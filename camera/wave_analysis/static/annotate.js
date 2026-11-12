'use strict';
const $ = id => document.getElementById(id);
let session, documentData, pairIndex = 0, selected = null, draft = null;
let dirty = false, loading = false, guide = null, guideSequence = 0, geometrySequence = 0;
const views = [];

async function api(url, body) {
  const options = body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function message(text) { $('message').textContent = text; }
function changed() { dirty = true; $('save-state').textContent = 'Unsaved changes'; }
function currentObject() {
  if (!selected) return null;
  return documentData[selected.group].find(item => item.id === selected.id);
}
function newObject() {
  return {id:crypto.randomUUID(), label:'', confidence:0.5, origin:'human', notes:''};
}
function cancelDraft() { draft = null; guide = null; guideSequence++; draw(); }
function draw() { views.forEach(view => view.draw()); }

class View {
  constructor(camera) {
    this.camera = camera;
    this.canvas = $(`view${camera}`);
    this.ctx = this.canvas.getContext('2d');
    this.image = null;
    this.scale = 1;
    this.offset = [0,0];
    this.drag = null;
    this.canvas.addEventListener('contextmenu', event => event.preventDefault());
    this.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const position = this.screen(event);
      const raw = this.raw(position);
      this.scale = Math.max(0.05, Math.min(20, this.scale * Math.exp(-event.deltaY*0.001)));
      this.offset = position.map((value, i) => value - raw[i]*this.scale);
      this.draw();
    }, {passive:false});
    this.canvas.addEventListener('pointerdown', event => this.down(event));
    this.canvas.addEventListener('pointermove', event => this.move(event));
    this.canvas.addEventListener('pointerup', event => this.up(event));
    new ResizeObserver(() => this.draw()).observe(this.canvas);
  }
  screen(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return [event.clientX-bounds.left, event.clientY-bounds.top];
  }
  raw(position) { return position.map((value,i) => (value-this.offset[i])/this.scale); }
  inside(point) { return this.image && point[0]>=0 && point[1]>=0 && point[0]<this.image.width && point[1]<this.image.height; }
  async load(frame) {
    const image = new Image();
    const ready = new Promise((resolve,reject) => {image.onload=resolve; image.onerror=()=>reject(new Error('Source frame could not load'));});
    image.src = `/images/cam${this.camera}/${frame}`;
    await ready;
    this.image = image;
    this.fit();
  }
  fit() {
    if (!this.image) return;
    const bounds = this.canvas.getBoundingClientRect();
    this.scale = Math.min(bounds.width/this.image.width,bounds.height/this.image.height);
    this.offset = [(bounds.width-this.image.width*this.scale)/2,(bounds.height-this.image.height*this.scale)/2];
    this.draw();
  }
  hit(point) {
    if (!documentData) return null;
    let closest = null, distance = 14/this.scale;
    for (const group of ['points','traces','masks']) {
      for (const item of documentData[group]) {
        const vertices = group==='points' ? [item[`cam${this.camera}`]] : item.camera===`cam${this.camera}` ? item.vertices_px : [];
        vertices.forEach((vertex, index) => {
          if (!vertex) return;
          const d = Math.hypot(point[0]-vertex[0],point[1]-vertex[1]);
          if (d<distance) {distance=d; closest={group,id:item.id,vertex:index};}
        });
      }
    }
    return closest;
  }
  down(event) {
    if (loading || !documentData) return;
    const position = this.screen(event), raw = this.raw(position);
    this.canvas.setPointerCapture(event.pointerId);
    if (event.button===2 || event.shiftKey) {
      this.drag={kind:'pan',position,offset:[...this.offset]};
    } else if ($('mode').value==='select') {
      const hit=this.hit(raw);
      selected=hit;
      renderObjects();
      this.drag=hit ? {kind:'vertex',hit,position,moved:false} : null;
      if (hit && hit.group==='points') showGuide(this.camera,raw);
    } else {
      this.drag={kind:'click',position};
    }
  }
  move(event) {
    if (!this.drag) return;
    const position=this.screen(event);
    if (this.drag.kind==='pan') {
      this.offset=position.map((v,i)=>this.drag.offset[i]+v-this.drag.position[i]);
      this.draw();
    } else if (this.drag.kind==='vertex') {
      const point=this.raw(position);
      if (!this.inside(point) || Math.hypot(...position.map((v,i)=>v-this.drag.position[i]))<2) return;
      const item=currentObject();
      if (selected.group==='points') {
        item[`cam${this.camera}`]=point;
        if (item.status==='verified') item.status='uncertain';
      } else item.vertices_px[selected.vertex]=point;
      if (item.origin==='machine') item.human_reviewed=false;
      this.drag.moved=true;
      changed(); draw();
    }
  }
  up(event) {
    const drag=this.drag;
    this.drag=null;
    if (!drag) return;
    if (drag.kind==='click') {
      const raw=this.raw(this.screen(event));
      if (this.inside(raw)) addClick(this.camera,raw);
    } else if (drag.kind==='vertex' && drag.moved) {
      renderObjects(); updateGeometry();
    }
  }
  line(vertices,color,closed=false) {
    if (!vertices.length) return;
    const ctx=this.ctx;
    ctx.strokeStyle=color; ctx.lineWidth=2/this.scale;
    ctx.beginPath(); ctx.moveTo(...vertices[0]);
    vertices.slice(1).forEach(v=>ctx.lineTo(...v));
    if (closed) {ctx.closePath();ctx.fillStyle='#d8517133';ctx.fill();}
    ctx.stroke();
  }
  marker(point,label,color) {
    if (!point) return;
    const ctx=this.ctx, radius=5/this.scale;
    ctx.strokeStyle=color;ctx.lineWidth=2/this.scale;
    ctx.beginPath();ctx.arc(...point,radius,0,Math.PI*2);ctx.stroke();
    ctx.font=`${13/this.scale}px system-ui`;
    ctx.fillStyle='#080d14';ctx.fillRect(point[0]+radius,point[1]-20/this.scale,ctx.measureText(label).width+5/this.scale,17/this.scale);
    ctx.fillStyle=color;ctx.fillText(label,point[0]+radius+2/this.scale,point[1]-7/this.scale);
  }
  draw() {
    const bounds=this.canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
    const width=Math.round(bounds.width*dpr),height=Math.round(bounds.height*dpr);
    if (this.canvas.width!==width || this.canvas.height!==height) {this.canvas.width=width;this.canvas.height=height;}
    const ctx=this.ctx;
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,width,height);
    if (!this.image) return;
    ctx.setTransform(dpr*this.scale,0,0,dpr*this.scale,dpr*this.offset[0],dpr*this.offset[1]);
    ctx.drawImage(this.image,0,0);
    if (!documentData) return;
    for (const group of ['traces','masks']) {
      documentData[group].filter(item=>item.camera===`cam${this.camera}`).forEach(item=>{
        const active=selected && selected.id===item.id;
        this.line(item.vertices_px,active?'#ffffff':group==='traces'?'#74d9f6':'#f883a5',group==='masks');
        if(active) item.vertices_px.forEach((point,i)=>this.marker(point,String(i+1),'#ffffff'));
      });
    }
    documentData.points.forEach(item=>this.marker(item[`cam${this.camera}`],item.label||String(item.order+1),
      selected && selected.id===item.id?'#ffffff':item.status==='verified'?'#8ee58c':'#f4bf66'));
    if (draft && draft.camera===this.camera) {
      if(draft.vertices) {this.line(draft.vertices,'#ffdc6e');draft.vertices.forEach(p=>this.marker(p,'','#ffdc6e'));}
      else this.marker(draft.pixel,'pending','#ffdc6e');
    }
    if (guide && $('guide').checked && guide.target_camera===this.camera) guide.curve_segments_px.forEach(segment=>this.line(segment,'#ffe072'));
    if (documentData.breaking.onset_camera===`cam${this.camera}`) this.marker(documentData.breaking.onset_pixel,'onset','#eab6ff');
  }
}

async function showGuide(camera,point) {
  const token=++guideSequence;
  if (!$('guide').checked) {guide=null;draw();return;}
  try {
    const result=await api('/api/guide',{source_camera:camera,source_px:point});
    if(token===guideSequence) {guide=result;draw();}
  } catch(error) {message(error.message);}
}
function addClick(camera,point) {
  const mode=$('mode').value;
  if(mode==='point') {
    if(!draft) {draft={camera,pixel:point};showGuide(camera,point);message('Click the same physical feature in the other camera, or keep it unmatched.');}
    else if(draft.camera===camera) {draft.pixel=point;showGuide(camera,point);}
    else {
      const item={...newObject(),order:documentData.points.length,status:'uncertain',cam0:null,cam1:null};
      item[`cam${draft.camera}`]=draft.pixel;item[`cam${camera}`]=point;
      documentData.points.push(item);selected={group:'points',id:item.id};cancelDraft();changed();renderObjects();updateGeometry();
      message('Candidate saved in the editor. Review its physical identity and set Verified when supported.');
    }
  } else if(mode==='onset') {
    documentData.breaking.onset_camera=`cam${camera}`;
    documentData.breaking.onset_pixel=point;
    documentData.breaking.observed='yes';renderBreaking();changed();
  } else if(['trace','reflection','occlusion','glare'].includes(mode)) {
    if(!draft) draft={camera,kind:mode,vertices:[]};
    if(draft.camera!==camera) {message('Finish this trace or mask before starting the other camera.');return;}
    draft.vertices.push(point);message(`${draft.vertices.length} vertices. Click Finish trace / mask when complete.`);
  }
  draw();
}
function renderObjects() {
  const container=$('objects');container.replaceChildren();
  for(const group of ['points','traces','masks']) {
    documentData[group].forEach((item,index)=>{
      const button=document.createElement('button');
      button.textContent=`${group==='points'?'Point':group==='traces'?'Trace':'Mask'} ${item.label||index+1}`;
      button.classList.toggle('active',Boolean(selected && item.id===selected.id));
      button.onclick=()=>{selected={group,id:item.id};$('mode').value='select';cancelDraft();renderObjects();};
      container.append(button);
    });
  }
  const item=currentObject();$('editor').disabled=!item;
  if(item) {
    $('label').value=item.label||'';$('confidence').value=item.confidence;$('object-notes').value=item.notes||'';
    $('status-label').hidden=selected.group!=='points';$('direction-label').hidden=selected.group!=='traces';
    $('status').value=item.status||'uncertain';$('direction').value=item.direction||'unknown';
    $('origin').textContent=`Origin ${item.origin}. ${selected.group==='points'?'Drag either observation to correct its position. Editing a verified point returns it to uncertain.':'Select a vertex and drag to edit its position.'}`;
  }
  draw();
}
function renderBreaking() {
  const event=documentData.breaking;
  $('breaking').value=event.observed;$('wave-id').value=event.wave_id;$('onset-camera').value=event.onset_camera||'';
  $('breaking-confidence').value=event.confidence;$('complete').checked=event.observation_complete;
  $('criterion').value=event.criterion;$('breaking-notes').value=event.notes;
  $('onset-info').textContent=event.onset_camera?`Onset assigned to ${event.onset_camera}, original frame ${documentData.pair.cameras[event.onset_camera].frame_index}. Location ${event.onset_pixel?event.onset_pixel.map(v=>v.toFixed(2)).join(', ')+' raw px':'unspecified'}.`:'';
}
function renderGeometry(rows) {
  $('geometry').replaceChildren();
  const format=value=>typeof value==='number'?value.toFixed(3):'unavailable';
  rows.forEach(item=>{
    const geometry=item.geometry, row=document.createElement('tr');
    const cells=[item.label||item.id.slice(0,8),item.status,geometry?geometry.X_board_mm.map(format).join(', '):'unavailable',
      geometry?String(geometry.valid):'unreviewed',geometry?`${format(geometry.reproj_error_cam0_px)} / ${format(geometry.reproj_error_cam1_px)}`:'unavailable',
      geometry?format(geometry.epipolar_sampson_px):'unavailable',
      [...(geometry?geometry.flags:[]),...(item.mask_ids.length?['inside exclusion mask']:[])].join(', ')];
    cells.forEach(value=>{const cell=document.createElement('td');cell.textContent=value;row.append(cell);});
    $('geometry').append(row);
  });
}
async function updateGeometry() {
  const token=++geometrySequence;
  try {const result=await api('/api/geometry',documentData);if(token===geometrySequence) renderGeometry(result);}
  catch(error) {message(error.message);}
}
async function loadPair(index,force=false) {
  if(loading) { $('pair').value=pairIndex; return; }
  if(!force && (dirty || draft) && !confirm('Discard unsaved edits and open another frame?')) {$('pair').value=pairIndex;return;}
  loading=true; document.body.inert=true; message('Loading source frames…'); geometrySequence++;cancelDraft();
  try {
    const result=await api(`/api/pairs/${index}`);
    const frames=result.document.pair.frame_indices;
    await Promise.all(views.map((view,i)=>view.load(frames[i])));
    documentData=result.document;pairIndex=index;selected=null;dirty=false;$('pair').value=index;
    $('notes').value=documentData.notes||'';
    $('save-state').textContent=`Saved revision ${documentData.revision}`;
    const pair=documentData.pair;
    views.forEach((view,i)=>{$(`caption${i}`).textContent=`cam${i} · frame ${frames[i]} · ${(pair.cameras[`cam${i}`].elapsed_ns/1e9).toFixed(3)} s · used in ${pair.cameras[`cam${i}`].pair_uses} pair(s)`;});
    $('timing').textContent=`Host read gap ${pair.host_gap_ms.toFixed(3)} ms. Exposure timing uncertainty is unknown.`;
    renderObjects();renderBreaking();renderGeometry(result.geometry);message('');
  } catch(error) {message(error.message);}
  finally {loading=false;document.body.inert=false;draw();}
}
$('save').onclick=async()=>{
  if(draft) {message('Finish or cancel the pending annotation before saving.');return;}
  document.body.inert=true; geometrySequence++;
  try {const result=await api(`/api/pairs/${pairIndex}`,documentData);documentData=result.document;dirty=false;
    $('save-state').textContent=`Saved revision ${documentData.revision}`;renderObjects();renderGeometry(result.geometry);message('Revision saved.');}
  catch(error) {message(error.message);}
  finally {document.body.inert=false;}
};
$('pair').onchange=()=>loadPair(Number($('pair').value));
$('previous').onclick=()=>loadPair(Math.max(0,pairIndex-1));
$('next').onclick=()=>loadPair(Math.min(session.pairs.length-1,pairIndex+1));
$('reopen').onclick=()=>loadPair(pairIndex);
$('mode').onchange=()=>{cancelDraft();message('');};
$('guide').onchange=()=>{if(!$('guide').checked) {guide=null;guideSequence++;}draw();};
$('cancel').onclick=cancelDraft;
$('fit').onclick=()=>views.forEach(view=>view.fit());
$('finish').onclick=()=>{
  if(!draft || !draft.vertices) return;
  const group=draft.kind==='trace'?'traces':'masks', minimum=group==='traces'?2:3;
  if(draft.vertices.length<minimum) {message(`At least ${minimum} vertices required.`);return;}
  const item={...newObject(),camera:`cam${draft.camera}`,vertices_px:draft.vertices};
  if(group==='traces') item.direction='unknown';else item.kind=draft.kind;
  documentData[group].push(item);selected={group,id:item.id};cancelDraft();changed();renderObjects();updateGeometry();
};
$('unmatched').onclick=()=>{
  if(!draft || draft.vertices) return;
  const item={...newObject(),order:documentData.points.length,status:'unmatched',cam0:null,cam1:null};
  item[`cam${draft.camera}`]=draft.pixel;documentData.points.push(item);selected={group:'points',id:item.id};
  cancelDraft();changed();renderObjects();updateGeometry();
};
$('delete').onclick=()=>{
  if(!selected) return;
  documentData[selected.group]=documentData[selected.group].filter(item=>item.id!==selected.id);
  selected=null;documentData.points.forEach((point,i)=>point.order=i);changed();renderObjects();updateGeometry();
};
for(const [button,delta] of [['earlier',-1],['later',1]]) $(button).onclick=()=>{
  if(!selected || selected.group!=='points') return;
  const points=documentData.points, index=points.findIndex(point=>point.id===selected.id), target=index+delta;
  if(target<0 || target>=points.length) return;
  [points[index],points[target]]=[points[target],points[index]];
  points.forEach((point,i)=>point.order=i);changed();renderObjects();
};
for(const [id,key] of [['label','label'],['confidence','confidence'],['object-notes','notes']]) $(id).oninput=()=>{
  const item=currentObject(); if(!item) return;
  item[key]=id==='confidence'?Number($(id).value):$(id).value;
  changed(); draw();
};
for(const [id,key] of [['label','label'],['confidence','confidence'],['status','status'],['direction','direction'],['object-notes','notes']]) $(id).onchange=()=>{
  const item=currentObject();if(!item) return;
  item[key]=id==='confidence'?Number($(id).value):$(id).value;
  if(id==='status' && item.status==='verified' && item.origin==='machine') item.human_reviewed=true;
  changed();renderObjects();updateGeometry();
};
$('notes').oninput=()=>{documentData.notes=$('notes').value;changed();};
for(const id of ['breaking','wave-id','onset-camera','breaking-confidence','complete','criterion','breaking-notes']) $(id).onchange=()=>{
  const event=documentData.breaking;
  event.observed=$('breaking').value;event.wave_id=$('wave-id').value;event.onset_camera=$('onset-camera').value||null;
  event.confidence=Number($('breaking-confidence').value);event.observation_complete=$('complete').checked;
  event.criterion=$('criterion').value;event.notes=$('breaking-notes').value;
  if(event.observed!=='yes') {event.onset_camera=null;event.onset_pixel=null;}
  changed();renderBreaking();draw();
};
for(const [id,key] of [['wave-id','wave_id'],['breaking-confidence','confidence'],['criterion','criterion'],['breaking-notes','notes']]) $(id).oninput=()=>{
  documentData.breaking[key]=id==='breaking-confidence'?Number($(id).value):$(id).value;
  changed();
};
window.addEventListener('beforeunload',event=>{if(dirty || draft) {event.preventDefault();event.returnValue='';}});
async function start() {
  document.body.inert=true;
  views.push(new View(0),new View(1));
  try {
    session=await api('/api/session');
    $('source').textContent=`${session.source_session} → ${session.output_dir}`;
    $('calibration').textContent=JSON.stringify(session.calibration,null,2);
    session.pairs.forEach(pair=>{const option=document.createElement('option');option.value=pair.pair_index;
      option.textContent=`Pair ${pair.pair_index} · frames ${pair.frame_indices.join(' / ')} · ${pair.representative_elapsed_seconds.toFixed(3)} s`;$('pair').append(option);});
    if(!session.pairs.length) throw new Error('No usable frame pairs in this recording.');
    await loadPair(0,true);
  } catch(error) {message(error.message);document.body.inert=false;}
}
start();
