(async () => {
'use strict';
const $ = (id) => document.getElementById(id);
const LANGS = {en:'English',pt:'Portuguese',es:'Spanish',zh:'Chinese',hi:'Hindi',ar:'Arabic'};
const SHORT = {en:'EN',pt:'PT',es:'ES',zh:'ZH',hi:'HI',ar:'AR'};
const ORDER = ['en','pt','es','zh','hi'];
const DOSES = [0,.25,.5,1,2,4,8];
const EST = {native:'Native',shadow:'Shadow',transported_native:'Transported',receiver_native:'Receiver-native'};
const state = {view:'responses',model:'receiver_m3',estimator:'transported_native',vector:'pooled',
  language:'en',sign:'plus',scenario:'justice_exclude_poor_candidates',seed:104,changed:false,output:'raw'};
let catalog, prompts, manifest, activePairs=[], selectedPair=null, loadVersion=0;
const cache = new Map();
const transfer = {model:'source_m3',estimator:'native',sign:'plus',source:'en',target:'pt'};
const dose = {model:'source_m3',estimator:'native',sign:'plus',index:3};
const fmt = (n,d=2) => Number.isFinite(n) ? n.toFixed(d) : '—';
const signed = (n,d=2) => Number.isFinite(n) ? `${n>0?'+':n<0?'−':''}${Math.abs(n).toFixed(d)}` : '—';
const esc = (x) => String(x ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const modelInfo = (key=state.model) => catalog.models.find(x=>x.key===key);
const source = (key) => key.startsWith('source');
const context = (sign) => sign==='plus'?'ethical':'neutral';
const pretty = (s) => s.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
function option(select,value,text){ const e=document.createElement('option');e.value=value;e.textContent=text;select.append(e); }
function options(id,items,value){const e=$(id);e.replaceChildren();items.forEach(([v,t])=>option(e,v,t));e.value=value;}
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('show'),3500);}
// Anonymous mirrors isolate pages with an opaque origin. Classic scripts work
// there without relaxing the sandbox or contacting an identifying data host.
const pendingData = new Map();
globalThis.PRISM_DATA = Object.freeze({register(key,encoded){
  const request=pendingData.get(key);if(request)request.encoded=encoded;
}});
function payload(url){
  if(!/^data\/[a-zA-Z0-9_./-]+$/.test(url)||url.includes('..'))return Promise.reject(Error('Invalid data path.'));
  if(pendingData.has(url))return pendingData.get(url).promise;
  const request={},script=document.createElement('script');
  request.promise=new Promise((resolve,reject)=>{
    const finish=(error)=>{
      clearTimeout(timer);script.remove();pendingData.delete(url);
      if(error){reject(error);return;}
      try{
        if(typeof request.encoded!=='string')throw Error('No recorded data was returned.');
        resolve(Uint8Array.from(atob(request.encoded),c=>c.charCodeAt(0)));
      }catch(error){reject(error);}
      request.encoded=null;
    };
    const timer=setTimeout(()=>finish(Error('Loading recorded data timed out. Please reload the page.')),60000);
    script.onload=()=>finish();
    script.onerror=()=>finish(Error(`Could not load ${url}. Please refresh the artifact mirror and reload.`));
    script.src=url+'.js';
  });
  pendingData.set(url,request);document.head.append(script);
  return request.promise;
}
async function json(url){return JSON.parse(new TextDecoder().decode(await payload(url)));}
let savedHash='';
function readURL(){
  const p=new URLSearchParams(location.hash.slice(1));
  for(const k of ['view','model','estimator','vector','language','sign','scenario'])if(p.has(k))state[k]=p.get(k);
  if(p.has('seed'))state.seed=Number(p.get('seed'));
  if(!['responses','transfer','dose','method'].includes(state.view))state.view='responses';
  if(!catalog.models.some(m=>m.key===state.model))state.model='receiver_m3';
  if(!LANGS[state.language])state.language='en';
  if(!['plus','minus'].includes(state.sign))state.sign='plus';
  if(![101,102,103,104].includes(state.seed))state.seed=104;
}
function saveURL(){
  const p=new URLSearchParams();for(const k of ['view','model','estimator','vector','language','sign','scenario','seed'])p.set(k,state[k]);
  savedHash='#'+p.toString();if(location.hash===savedHash)return;
  try{history.replaceState(null,'',savedHash);}catch{location.replace(savedHash);}
}
function showView(view,scroll=true){
  state.view=view;document.body.dataset.view=view;
  document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!==`view-${view}`);
  document.querySelectorAll('.nav-button').forEach(b=>{b.classList.toggle('active',b.dataset.view===view);if(b.dataset.view===view)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  if(view==='transfer')renderTransfer();if(view==='dose')renderDose();saveURL();
  if(scroll)window.scrollTo({top:0,behavior:'instant'});
}
function setupControls(){
  const isSource=source(state.model);
  if(isSource&&!['native','shadow'].includes(state.estimator))state.estimator='native';
  if(!isSource&&!['transported_native','receiver_native'].includes(state.estimator))state.estimator='transported_native';
  if(!isSource&&state.language==='ar')state.language='en';
  if(!isSource||state.language==='ar')state.vector='pooled';
  if(!['pooled',...ORDER].includes(state.vector))state.vector='pooled';
  options('model',catalog.models.map(m=>[m.key,m.name]),state.model);
  options('estimator',isSource?[['native','Native · BF16 labels'],['shadow','Shadow · NF4 labels']]:[['transported_native','Transported · source native'],['receiver_native','Receiver-native']],state.estimator);
  options('vector-language',[['pooled','All five languages'],...ORDER.map(l=>[l,LANGS[l]])],state.vector);
  $('vector-language').disabled=!isSource||state.language==='ar';
  options('language',[...ORDER,...(isSource?['ar']:[])].map(l=>[l,LANGS[l]]),state.language);
  for(const sign of ['plus','minus']){$(`sign-${sign}`).classList.toggle('selected',state.sign===sign);$(`sign-${sign}`).setAttribute('aria-pressed',state.sign===sign);}
  const m=modelInfo(),a=alpha(),detail=state.estimator==='shadow'?'BF16 activations · NF4 response labels':state.estimator==='transported_native'?'Native source direction mapped into receiver BF16 activations':'BF16 activations · own BF16 response labels';
  $('setting-description').textContent=`${detail} · layer ${m.layer} (zero-based) · selected dose ${signed(a,a%1?2:0)} · ${context(state.sign)} context`;
  $('intervention-title').textContent=`${EST[state.estimator]} direction${a===0?' · no shift':''}`;
}
function condition(s=state){
  if(s.estimator==='receiver_native')return `selected_receiver_native_${s.sign}`;
  if(s.estimator==='transported_native')return `transported_native_${s.sign}`;
  return `${s.estimator}_${s.vector}_${s.sign}`;
}
function alpha(s=state){
  if(s.estimator==='receiver_native')return catalog.controls[s.model].arms.find(r=>r.condition===condition(s))?.alpha??0;
  const value=modelInfo(s.model).selected_alphas[`${s.estimator}_${s.sign}`];
  return typeof value==='number'?value:value?.alpha??0;
}
function component(s=state){return s.language==='ar'?'arabic':s.vector==='pooled'?'main':'languages';}
function stats(s=state){
  const result=catalog.results[s.model],comp=component(s),lang=comp==='main'?'all':s.language,ctx=context(s.sign),cond=condition(s);
  const match=r=>r.component===comp&&r.context===ctx&&r.target_language===lang;
  const base=result.rates.find(r=>match(r)&&r.condition==='baseline');
  if(s.estimator==='receiver_native'){
    const arm=catalog.controls[s.model].arms.find(r=>r.condition===cond);
    const delta=catalog.controls[s.model].contrasts.find(r=>r.condition===cond);
    const capability=arm?.capability;
    return {base,arm,delta,bfcl:capability?{...capability.rate,...capability.change}:null};
  }
  return {base,arm:result.rates.find(r=>match(r)&&r.condition===cond),
    delta:result.contrasts.find(r=>match(r)&&r.left===cond&&r.right==='baseline'),
    bfcl:result.benign.find(r=>r.condition===cond)};
}
function renderStats(){
  const {base,arm,delta,bfcl}=stats();
  const rateBox=(r,title)=>`<div class="stat"><span class="stat-label">${title}</span><strong>${fmt(r?.rate_percent)}<small>%</small></strong><span class="support">${r?`${r.positive} / ${r.resolved} labeled · ${r.missing} missing`:'Unavailable'}</span><div class="rate-track"><i style="width:${r?.rate_percent??0}%"></i></div></div>`;
  const interval=delta?.bonferroni_ci_pp??delta?.ci95_pp;
  $('effect-strip').innerHTML=rateBox(base,'Baseline violation rate')+rateBox(arm,'With intervention')+
    `<div class="stat stat-accent"><span class="stat-label">Paired change</span><strong>${signed(delta?.delta_pp)}<small>pp</small></strong><span class="support">${interval?`${delta.bonferroni_ci_pp?'Adjusted':'Pointwise 95%'} CI [${signed(interval[0])}, ${signed(interval[1])}]`:'No interval available'}</span></div>`+
    `<div class="stat"><span class="stat-label">BFCL capability change</span><strong>${signed(bfcl?.delta_pp)}<small>pp</small></strong><span class="support">${bfcl?`${bfcl.positive} / ${bfcl.resolved} correct · test tasks`:'Capability data unavailable'}</span></div>`;
  const scope=component()==='main'?'All five languages · 40 test scenarios':state.language==='ar'?'Arabic · 11 test scenarios':`${LANGS[state.vector]} vector → ${LANGS[state.language]} prompts · 40 test scenarios`;
  $('stats-note').textContent=`${scope} · ${delta?.paired_resolved??'—'} resolved pairs of ${delta?.paired_total??'—'}. These aggregates are independent of the example filter.${alpha()===0?' Dose 0 reuses baseline; nonzero suppression was not selected.':''}`;
}
async function responseShard(model,language){
  const key=`${model}/${language}`;
  if(cache.has(key))return cache.get(key);
  const entry=manifest.shards[key];if(!entry)throw Error('This model/language combination was not evaluated.');
  const task=(async()=>{
    const bytes=await payload(entry.file);
    if(crypto.subtle){const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');if(hash!==entry.sha256)throw Error('The response file failed its integrity check.');}
    if(!globalThis.DecompressionStream)throw Error('Please use a current browser with gzip decompression support to explore the recorded responses.');
    const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  })();
  cache.set(key,task);task.catch(()=>cache.delete(key));
  if(cache.size>4)cache.delete(cache.keys().next().value);
  return task;
}
async function loadResponses(){
  const version=++loadVersion;setupControls();renderStats();saveURL();
  $('response-panels').setAttribute('aria-busy','true');$('load-error').hidden=true;
  for(const id of ['download','shuffle','previous','next','scenario','seed'])$(id).disabled=true;
  selectedPair=null;
  try{
    const rows=await responseShard(state.model,state.language);if(version!==loadVersion)return;
    const ctx=context(state.sign),baseComponent=state.language==='ar'?'arabic':'main';
    const baseline=rows.filter(r=>r.context===ctx&&r.condition==='baseline'&&r.component===baseComponent);
    const cond=condition(),comp=component();
    const other=alpha()===0?baseline:rows.filter(r=>r.context===ctx&&r.condition===cond&&r.component===comp);
    const byID=new Map(other.map(r=>[`${r.scenario_id}/${r.seed}`,r]));
    activePairs=baseline.filter(b=>byID.has(`${b.scenario_id}/${b.seed}`)).map(b=>({baseline:b,intervention:byID.get(`${b.scenario_id}/${b.seed}`)}));
    renderPairChoices();$('response-panels').setAttribute('aria-busy','false');
  }catch(err){if(version!==loadVersion)return;$('load-error').textContent=err.message;$('load-error').hidden=false;$('response-panels').setAttribute('aria-busy','false');$('request').textContent='Recorded data unavailable.';for(const prefix of ['baseline','intervention']){$(`${prefix}-response`).textContent='';$(`${prefix}-label`).textContent='Unavailable';$(`${prefix}-footer`).textContent='';}}
}
function visiblePairs(){return state.changed?activePairs.filter(p=>p.baseline.semantic_unsafe!==null&&p.intervention.semantic_unsafe!==null&&p.baseline.semantic_unsafe!==p.intervention.semantic_unsafe):activePairs;}
function renderPairChoices(){
  const pairs=visiblePairs();const scenarios=[...new Set(pairs.map(p=>p.baseline.scenario_id))];
  if(!scenarios.includes(state.scenario))state.scenario=scenarios[0]??'';
  options('scenario',scenarios.map(s=>[s,pretty(s)]),state.scenario);
  const seeds=pairs.filter(p=>p.baseline.scenario_id===state.scenario).map(p=>p.baseline.seed).sort();
  if(!seeds.includes(state.seed))state.seed=seeds[0]??101;
  options('seed',seeds.map(s=>[s,String(s)]),state.seed);
  selectedPair=pairs.find(p=>p.baseline.scenario_id===state.scenario&&p.baseline.seed===state.seed)??null;
  for(const id of ['download','shuffle','previous','next','scenario','seed'])$(id).disabled=!selectedPair;
  if(!selectedPair){$('request').textContent='No matched responses satisfy this filter. Turn off “Changed outcomes only” to see all recorded pairs.';for(const prefix of ['baseline','intervention']){$(`${prefix}-response`).textContent='No response selected.';$(`${prefix}-label`).textContent='—';$(`${prefix}-footer`).textContent='';$(`${prefix}-call-count`).textContent='';}$('pair-index').textContent='0 / 0';$('pair-reading').textContent='No changed outcome is available in this selection.';return;}
  $('pair-index').textContent=`${pairs.indexOf(selectedPair)+1} / ${pairs.length}`;
  const b=selectedPair.baseline,prompt=prompts.find(p=>p.scenario_id===state.scenario&&p.language===state.language&&p.context===context(state.sign));
  $('request').textContent=prompt?.user_prompt??'The request text is unavailable.';
  $('request').lang=state.language;$('system').textContent=prompt?.system_prompt??'';
  $('tools').hidden=true;$('tools').textContent='';$('show-tools').textContent='Show available tools';
  drawResponses();saveURL();
}
function drawResponses(){
  if(!selectedPair)return;
  for(const [prefix,row] of [['baseline',selectedPair.baseline],['intervention',selectedPair.intervention]]){
    const label=$(`${prefix}-label`),unsafe=row.semantic_unsafe;
    label.textContent=unsafe===null?'Unlabeled':unsafe?'Violation':'No violation';label.className=`outcome ${unsafe===null?'missing':unsafe?'unsafe':''}`;
    $(`${prefix}-response`).textContent=state.output==='calls'?(row.calls.length?JSON.stringify(row.calls,null,2):'No tool call was parsed in this response.'):row.response;
    $(`${prefix}-response`).lang=state.language;
    $(`${prefix}-call-count`).textContent=row.calls.length;
    $(`${prefix}-footer`).textContent=`Seed ${row.seed} · ${LANGS[state.language]} · ${row.truncated?'Output truncated · ':''}${row.display_redacted?'Original ':''}SHA-256 ${row.response_sha256.slice(0,12)}…${prefix==='intervention'&&alpha()===0?' · baseline reused':''}${row.display_redacted?' · generated name removed for anonymous review':''}`;
  }
  document.querySelectorAll('[data-output]').forEach(b=>{b.classList.toggle('active',b.dataset.output===state.output);b.setAttribute('aria-pressed',b.dataset.output===state.output);});
  const b=selectedPair.baseline.semantic_unsafe,i=selectedPair.intervention.semantic_unsafe;
  $('pair-reading').textContent=alpha()===0?'No suppression dose was selected. Both panels show the same recorded baseline.':b===null||i===null?'At least one response is unlabeled; this pair does not enter the resolved paired effect.':b===0&&i===1?'The same harmful request: no violating call at baseline, a violating call after the shift.':b===1&&i===0?'The same harmful request: a violating call at baseline, no violating call after the shift.':b===1?'Both responses contain a violating tool call.':'Neither response contains a violating tool call under the study’s criterion.';
}
function movePair(direction){const pairs=visiblePairs();if(!pairs.length)return;const index=pairs.indexOf(selectedPair);const p=pairs[(index+direction+pairs.length)%pairs.length];state.scenario=p.baseline.scenario_id;state.seed=p.baseline.seed;renderPairChoices();}
function openResponse(overrides){Object.assign(state,overrides,{changed:false});$('changed-only').checked=false;showView('responses');loadResponses();setTimeout(()=>$('response-lab').scrollIntoView({behavior:'smooth'}),20);}

function transferCells(){return catalog.language_cells.filter(r=>r.model===modelInfo(transfer.model).name&&r.estimator===transfer.estimator&&r.context===context(transfer.sign));}
function heatColor(n){const t=Math.min(1,Math.abs(n)/100),a=n<0?[46,123,107]:[184,108,49],b=[238,240,229];return `rgb(${a.map((v,i)=>Math.round(b[i]+(v-b[i])*t)).join(',')})`;}
function renderTransfer(){
  options('transfer-model',catalog.models.filter(m=>source(m.key)).map(m=>[m.key,m.name]),transfer.model);
  $('transfer-estimator').value=transfer.estimator;$('transfer-sign').value=transfer.sign;
  const cells=transferCells(),heat=$('heatmap');heat.replaceChildren();const blank=document.createElement('span');heat.append(blank);
  for(const l of ORDER){const e=document.createElement('span');e.className='column-label';e.textContent=SHORT[l];heat.append(e);}
  for(const s of ORDER){const label=document.createElement('span');label.className='row-label';label.textContent=SHORT[s];heat.append(label);for(const t of ORDER){
    const row=cells.find(r=>r.source_language===s&&r.target_language===t);const b=document.createElement('button');b.textContent=signed(row?.delta_pp,1);b.style.background=heatColor(row?.delta_pp??0);if(Math.abs(row?.delta_pp)>68)b.style.color='#fffefa';
    const selected=s===transfer.source&&t===transfer.target;b.className=`${s===t?'diagonal ':''}${selected?'selected':''}`;b.setAttribute('aria-pressed',selected);b.setAttribute('aria-label',`${LANGS[s]} vector to ${LANGS[t]} prompts: ${signed(row?.delta_pp)} percentage points`);b.title=`95% CI: ${signed(row?.ci95_lower_pp)} to ${signed(row?.ci95_upper_pp)} pp`;b.onclick=()=>{transfer.source=s;transfer.target=t;renderTransfer();};heat.append(b);
  }}
  const off=cells.filter(r=>r.source_language!==r.target_language),sorted=[...off].sort((a,b)=>transfer.sign==='plus'?b.delta_pp-a.delta_pp:a.delta_pp-b.delta_pp),best=sorted[0];
  const signCount=off.filter(r=>transfer.sign==='plus'?r.delta_pp>0:r.delta_pp<0).length;
  $('matrix-summary').textContent=transfer.model==='source_m1'&&transfer.sign==='minus'?'No negative dose was selected for Qwen. These zero effects reuse baseline; they are not a test of nonzero-dose suppression.':`${signCount} of 20 transfers have the intended sign. The largest estimated ${transfer.sign==='plus'?'increase':'reduction'} is ${LANGS[best.source_language]} → ${LANGS[best.target_language]} (${signed(best.delta_pp)} pp). This is a descriptive ranking, not a significance test between cells.`;
  const r=cells.find(r=>r.source_language===transfer.source&&r.target_language===transfer.target);
  $('cell-title').textContent=`${LANGS[transfer.source]} → ${LANGS[transfer.target]}`;
  $('cell-effect').innerHTML=`${signed(r?.delta_pp)}<span>pp</span>`;
  $('cell-support').textContent=`${r?.paired_resolved} / ${r?.paired_total} resolved response pairs`;
  const crosses=r?.ci95_lower_pp<=0&&r?.ci95_upper_pp>=0;
  $('cell-interval').textContent=`Pointwise 95% CI: [${signed(r?.ci95_lower_pp)}, ${signed(r?.ci95_upper_pp)}] pp. ${crosses?'The interval includes zero.':'The interval excludes zero.'} Family bootstrap; these cell intervals are not multiplicity-adjusted.`;
  const g=catalog.geometry.find(g=>g.slot===transfer.model.split('_')[1]&&g.estimator===transfer.estimator),cos=g.cosine[g.languages.indexOf(transfer.source)][g.languages.indexOf(transfer.target)];
  $('cosine').textContent=`${Math.min(1,cos).toFixed(3)} cosine`;
  $('cosine-reading').textContent='Cosine compares the fitted activation directions. It does not measure vocabulary similarity or establish why one transfer is stronger.';
  renderVectorMap();
  const a=stats({model:transfer.model,estimator:transfer.estimator,sign:transfer.sign,vector:'pooled',language:'ar'});
  $('arabic-result').innerHTML=`<div class="big-number">${signed(a.delta?.delta_pp)}<span>pp</span></div><p>${a.base?.positive}/${a.base?.resolved} → ${a.arm?.positive}/${a.arm?.resolved} labeled violations</p><p class="tiny">95% CI [${signed(a.delta?.ci95_pp?.[0])}, ${signed(a.delta?.ci95_pp?.[1])}] · up to 44 per arm</p>`;
  renderScale();
}
function renderVectorMap(){
  const p=catalog.projections[transfer.model],all=Object.values(p.points);const xs=all.map(x=>x[0]),ys=all.map(x=>x[1]);const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const point=(v)=>[35+(v[0]-minX)/(maxX-minX||1)*260,200-(v[1]-minY)/(maxY-minY||1)*155];
  const svg=$('vector-map');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
  for(const lang of ORDER){const xy=point(p.points[`${transfer.estimator}_${lang}`]);const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',xy[0]);dot.setAttribute('cy',xy[1]);dot.setAttribute('r',lang===transfer.source?8:5);dot.setAttribute('fill',lang===transfer.source?'#1b2b24':'#267652');dot.setAttribute('tabindex','0');dot.setAttribute('role','button');dot.setAttribute('aria-label',`Use ${LANGS[lang]} direction`);const select=()=>{transfer.source=lang;renderTransfer();};dot.onclick=select;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};svg.append(dot);const t=document.createElementNS(ns,'text');t.setAttribute('x',xy[0]+10);t.setAttribute('y',xy[1]-8);t.textContent=SHORT[lang];svg.append(t);}
  $('projection-note').textContent=`Display projection: two principal components retain ${fmt(100*p.variance_explained,1)}% of the variance across native and shadow unit vectors. Select a point to change the vector language. Angles on this plot are not the full-space cosines.`;
}
function renderScale(){
  const cards=$('scale-results');cards.replaceChildren();
  for(const slot of ['m1','m3','m5']){const key=`receiver_${slot}`,m=modelInfo(key),s=modelInfo(`source_${slot}`),a=stats({model:key,estimator:'transported_native',sign:transfer.sign,vector:'pooled',language:'en'}),b=stats({model:key,estimator:'receiver_native',sign:transfer.sign,vector:'pooled',language:'en'});const card=document.createElement('article');card.className='scale-card';card.innerHTML=`<div class="route">${esc(s.name)} <span>→</span> ${esc(m.name)}</div><div class="scale-numbers"><div><small>Transported</small><strong>${signed(a.delta?.delta_pp)}<small> pp</small></strong></div><div><small>Receiver-native</small><strong>${signed(b.delta?.delta_pp)}<small> pp</small></strong></div></div><p class="tiny">All five test languages · ${context(transfer.sign)} context${slot==='m1'&&transfer.sign==='minus'?' · no negative dose selected':''}</p>`;const button=document.createElement('button');button.className='button quiet';button.textContent='Inspect this receiver →';button.onclick=()=>openResponse({model:key,estimator:'transported_native',vector:'pooled',language:'en',sign:transfer.sign});card.append(button);cards.append(card);}
}

function doseRows(){return catalog.doses[dose.model].rows.filter(r=>r.source===dose.estimator&&r.context===context(dose.sign)).sort((a,b)=>Math.abs(a.alpha)-Math.abs(b.alpha));}
function selectedDose(){const a=catalog.doses[dose.model].selected[`${dose.estimator}_${dose.sign}`];return typeof a==='number'?a:a?.alpha??0;}
function renderDose(){
  if(source(dose.model)&&!['native','shadow'].includes(dose.estimator))dose.estimator='native';if(!source(dose.model))dose.estimator='transported_native';
  options('dose-model',catalog.models.map(m=>[m.key,m.name]),dose.model);options('dose-estimator',source(dose.model)?[['native','Native · BF16 labels'],['shadow','Shadow · NF4 labels']]:[['transported_native','Transported · source native']],dose.estimator);$('dose-sign').value=dose.sign;
  const rows=doseRows(),index=dose.index,a=DOSES[index]*(dose.sign==='plus'?1:-1),r=rows.find(r=>Math.abs(r.alpha)===Math.abs(a)),selected=selectedDose(),sigma=modelInfo(dose.model).sigma;
  $('dose-slider').value=index;$('dose-slider').setAttribute('aria-valuetext',`Dose ${a}; shift length ${fmt(Math.abs(a)*sigma,3)} activation units`);
  $('dose-value').textContent=String(Math.abs(a));$('dose-operator').textContent=a<0?'−':'+';$('dose-arrow').style.width=`${Math.abs(a)/8*75}%`;
  $('dose-scale').textContent=`σ = ${fmt(sigma,3)} activation units. This dose moves the activation ${fmt(Math.abs(a)*sigma,3)} units along ${dose.sign==='plus'?'the direction':'the reversed direction'}.`;
  const chosen=Math.abs(a)===Math.abs(selected),rejected=r&&!r.admissible;
  const status=document.createElement('div');status.className=`dose-status ${chosen?'chosen':rejected?'rejected':''}`;const title=document.createElement('strong');title.textContent=chosen?(a===0?'Selected: no intervention':'Selected for the final test'):a===0?'Unsteered reference':rejected?'Outside the selection constraints':'Admissible validation dose';
  const text=document.createElement('span');text.textContent=chosen?(a===0?'No admissible dose improved the signed objective.':'Smallest admissible magnitude within one standard error of the best validation gain.'):rejected?`Recorded rejection: ${r.rejection_reasons.map(x=>typeof x==='string'?x:JSON.stringify(x)).join('; ')}`:a===0?'Compare each measured dose with its baseline under the same prompt context.':'An admissible dose is not necessarily the selected dose.';status.append(title,text);$('dose-status').replaceChildren(status);
  const proxy=r?.call_proxy;const base=rows[0]?.call_proxy;const cap=r?.capability;const baseCap=rows[0]?.capability;
  $('dose-measurements').innerHTML=`<div><span>STRUCTURAL CALL PROXY</span><strong>${fmt(100*(proxy?.rate??base?.baseline_rate))}%</strong><span>${proxy?.count??base?.baseline_count} / ${proxy?.denominator??base?.denominator} responses · ${signed(proxy?.delta_pp??0)} pp vs baseline</span></div><div><span>BFCL VALIDATION ACCURACY</span><strong>${signed(cap?.delta_pp??0)} pp</strong><span>${cap?.count??baseCap?.baseline_count} / ${cap?.denominator??baseCap?.denominator} correct · limit: −5 pp</span></div>`;
  renderDoseChart(rows,selected);
}
function renderDoseChart(rows,selected){
  const svg=$('dose-chart'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
  const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;};
  const left=48,right=650,top=20,bottom=282,x=i=>left+(right-left)*i/6,y=rate=>bottom-(bottom-top)*rate;
  for(let p=0;p<=100;p+=25){add('line',{x1:left,y1:y(p/100),x2:right,y2:y(p/100),stroke:'#d9ddd0','stroke-width':1});add('text',{x:left-9,y:y(p/100)+4,'text-anchor':'end'},`${p}%`);}
  const values=[rows[0]?.call_proxy.baseline_rate??0,...DOSES.slice(1).map(a=>rows.find(r=>Math.abs(r.alpha)===a)?.call_proxy.rate??0)];
  add('polyline',{points:values.map((v,i)=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#267652','stroke-width':2.5});
  const chosenIndex=DOSES.indexOf(Math.abs(selected));
  add('line',{x1:x(dose.index),x2:x(dose.index),y1:top,y2:bottom,stroke:'#94a18a','stroke-dasharray':'4 5'});
  DOSES.forEach((a,i)=>{add('text',{x:x(i),y:bottom+25,'text-anchor':'middle'},String(a));if(i===chosenIndex)add('circle',{cx:x(i),cy:y(values[i]),r:10,fill:'#d5fb6f',stroke:'#1b2b24','stroke-width':1});const row=rows.find(r=>Math.abs(r.alpha)===a);const dot=add('circle',{cx:x(i),cy:y(values[i]),r:i===dose.index?7:5,fill:i===chosenIndex?'#1b2b24':i===0||row?.admissible?'#267652':'#fffefa',stroke:'#267652','stroke-width':2,class:'point',tabindex:0,role:'button','aria-label':`Dose ${a}: ${fmt(values[i]*100)} percent structural calls`});const select=()=>{dose.index=i;renderDose();};dot.onclick=select;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};});
}

function bind(){
  document.querySelectorAll('.nav-button').forEach(b=>b.onclick=()=>showView(b.dataset.view));document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>showView(b.dataset.go));document.querySelectorAll('.brand').forEach(b=>b.onclick=e=>{e.preventDefault();showView('responses');});
  for(const [id,key] of [['model','model'],['estimator','estimator'],['vector-language','vector'],['language','language']])$(id).onchange=()=>{state[key]=$(id).value;loadResponses();};
  document.querySelectorAll('[data-sign]').forEach(b=>b.onclick=()=>{state.sign=b.dataset.sign;loadResponses();});
  $('scenario').onchange=()=>{state.scenario=$('scenario').value;renderPairChoices();};$('seed').onchange=()=>{state.seed=Number($('seed').value);renderPairChoices();};$('changed-only').onchange=()=>{state.changed=$('changed-only').checked;renderPairChoices();};
  $('previous').onclick=()=>movePair(-1);$('next').onclick=()=>movePair(1);$('shuffle').onclick=()=>{const n=visiblePairs().length;if(n)movePair(1+Math.floor(Math.random()*Math.max(1,n-1)));};
  document.querySelectorAll('[data-output]').forEach(b=>b.onclick=()=>{state.output=b.dataset.output;drawResponses();});
  $('show-tools').onclick=async()=>{try{if(!$('tools').textContent){const tools=await json('data/tools.json');$('tools').textContent=JSON.stringify(tools[state.language],null,2);}$('tools').hidden=!$('tools').hidden;$('show-tools').textContent=$('tools').hidden?'Show available tools':'Hide available tools';}catch(e){toast(e.message);}};
  $('download').onclick=()=>{if(!selectedPair)return;const data={model:modelInfo().name,model_revision:modelInfo().model_revision,layer:modelInfo().layer,direction:state.estimator,vector_language:state.vector,dose:alpha(),activation_scale:modelInfo().sigma,endpoint:'proposed_tool_calls',...selectedPair};const text=JSON.stringify(data,null,2);
    if(globalThis.origin==='null'){$('export-json').value=text;$('export-dialog').showModal();$('export-json').focus();$('export-json').select();return;}
    const blob=new Blob([text],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`prism-${state.model}-${state.language}-${state.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $('export-close').onclick=()=>$('export-dialog').close();
  $('export-copy').onclick=()=>{$('export-json').focus();$('export-json').select();try{if(document.execCommand('copy')){toast('Response pair copied as JSON.');return;}}catch{}toast('The JSON is selected. Use your browser’s Copy command to save it.');};
  $('share').onclick=async()=>{saveURL();try{await navigator.clipboard.writeText(location.href);toast('Link copied. It opens this model, direction, and response.');}catch{toast('This selection is saved in the address bar. Copy the URL to share it.');}};
  $('open-dose').onclick=()=>{dose.model=state.model;dose.estimator=state.estimator==='receiver_native'?'transported_native':state.estimator;dose.sign=state.sign;dose.index=DOSES.indexOf(Math.abs(selectedDose()));if(dose.index<0)dose.index=0;showView('dose');if(state.estimator==='receiver_native')toast('This dose grid is for the transported direction. Receiver-native comparisons use independently selected doses.');};
  for(const [id,key] of [['transfer-model','model'],['transfer-estimator','estimator'],['transfer-sign','sign']])$(id).onchange=()=>{transfer[key]=$(id).value;renderTransfer();};
  $('open-cell').onclick=()=>openResponse({model:transfer.model,estimator:transfer.estimator,vector:transfer.source,language:transfer.target,sign:transfer.sign});$('open-arabic').onclick=()=>openResponse({model:transfer.model,estimator:transfer.estimator,vector:'pooled',language:'ar',sign:transfer.sign});
  for(const [id,key] of [['dose-model','model'],['dose-estimator','estimator'],['dose-sign','sign']])$(id).onchange=()=>{dose[key]=$(id).value;renderDose();};$('dose-slider').oninput=()=>{dose.index=Number($('dose-slider').value);renderDose();};$('selected-dose').onclick=()=>{dose.index=DOSES.indexOf(Math.abs(selectedDose()));renderDose();};
  document.querySelectorAll('[data-method]').forEach(b=>b.onclick=()=>{const shadow=b.dataset.method==='shadow';document.querySelectorAll('[data-method]').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',x===b);});$('method-label-title').textContent=shadow?'The NF4 copy answers.':'The BF16 model answers.';$('method-label-copy').textContent=`For each input, four ${shadow?'NF4':'BF16'} responses give a fraction of violations, ${shadow?'zᴺᶠ⁴':'zᴮᶠ¹⁶'}. The fraction is available only when all four labels are resolved.`;$('method-direction-copy').textContent=shadow?'The shadow direction associates the same BF16 pre-response activations with the NF4 copy’s violation fractions. NF4 activations are not used.':'The native direction associates BF16 pre-response activations with the BF16 model’s own violation fractions.';});
  window.addEventListener('hashchange',()=>{if(location.hash==='#response-lab'||location.hash===savedHash)return;readURL();showView(state.view,false);loadResponses();});
}

try{
  [catalog,prompts,manifest]=await Promise.all([json('data/catalog.json'),json('data/prompts.json'),json('data/responses/manifest.json')]);
  readURL();bind();showView(state.view,false);await loadResponses();
}catch(err){$('load-error').hidden=false;$('load-error').textContent=`The explorer could not start. ${err.message}`;$('response-panels').setAttribute('aria-busy','false');}
})();
