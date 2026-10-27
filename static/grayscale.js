// Use the same canvas, brushes, line tools, SAM tools and undo history as RGB.
// All saves go to drafts. Export publishes a complete set in grayscale_mask_export/.
let grayBusy = false, grayReady = false, grayDirty = false, grayPending = 0;
let grayMaskLoad = Promise.resolve();

function grayStatus(){
  document.getElementById('draftStatus').textContent =
    `${grayPending} frame(s) awaiting Export` + (grayDirty ? ' · current frame edited' : '');
}
async function grayJson(url, body){
  const r = await fetch(url, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json();
  if(!r.ok || d.error) throw new Error(d.error || `Request failed: ${r.status}`);
  return d;
}
async function grayRun(action){
  if(grayBusy || down) return;
  grayBusy = true;
  document.getElementById('exportBtn').disabled = true;
  try { await action(); await grayMaskLoad; }
  catch(e){ document.getElementById('clickStatus').textContent = e.message; }
  finally { grayBusy = false; document.getElementById('exportBtn').disabled = false; grayStatus(); }
}
function grayImage(src){
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('Could not load image or mask; edits have not been exported.'));
    im.src = src;
  });
}
loadMaskFromUrl = function(url){
  grayMaskLoad = grayImage(url).then(im => {
    ensureMaskSize();
    if(im.width !== maskCanvas.width || im.height !== maskCanvas.height)
      throw new Error('Mask dimensions do not match the keyframe.');
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.drawImage(im, 0, 0);
  });
  return grayMaskLoad;
};
stampMaskFromUrl = function(url, subtract){
  grayMaskLoad = grayImage(url).then(im => {
    ensureMaskSize();
    if(im.width !== maskCanvas.width || im.height !== maskCanvas.height)
      throw new Error('Mask dimensions do not match the keyframe.');
    maskCtx.globalCompositeOperation = subtract ? 'destination-out' : 'source-over';
    maskCtx.drawImage(im, 0, 0);
    maskCtx.globalCompositeOperation = 'source-over';
  });
  return grayMaskLoad;
};

maskEditable = () => grayReady && !grayBusy && !frames[idx]?.excluded;
const grayPushUndo = pushUndo;
pushUndo = function(){ grayPushUndo(); grayDirty = true; grayStatus(); };
const grayUndo = undoMask;
undoMask = function(){
  if(!maskEditable() || !undoStack.length) return;
  grayUndo(); grayDirty = true; grayStatus();
};
const grayClick = handleClick, graySmart = refineSmart;
handleClick = e => maskEditable() && grayRun(() => tool === 'msmart' ? grayMultiSmart(e) : grayClick(e));
refineSmart = () => maskEditable() && grayRun(() => graySmart());

async function grayMultiSmart(e){
  const point = evToNatural(e);
  if(!inFrame(...point)) return;
  document.getElementById('clickStatus').textContent = 'Multi smart brush: refining selected blob…';
  const d = await grayJson('/api/multi_smart', {
    id: reviewObj, frame: frames[idx].name, point, shift: e.shiftKey,
    mask: maskCanvas.toDataURL('image/png')});
  if(d.added || d.removed){
    pushUndo();
    await loadMaskFromUrl(d.mask);
    frames[idx].px = d.px; updateLabel();
  }
  document.getElementById('clickStatus').textContent = d.excluded ? 'Scene-change-excluded frame: no changes'
    : d.nothing_near_click ? 'Multi smart brush: nothing near click'
    : `Multi smart brush: +${d.added} / −${d.removed} px · ${d.holes_filled} small holes filled`
      + (d.sam_rejected ? ' · sam rejected (hole fill only)' : '')
      + (e.shiftKey ? ' · no trimming' : '');
}

async function grayLoadFrame(){
  grayReady = false; editing = true; pos = []; neg = []; endLine(); resetUndo();
  setEditUI(true); showMask(false);
  const f = frames[idx];
  await new Promise((resolve, reject) => {
    view.onload = () => { ensureMaskSize(); resolve(); };
    view.onerror = () => reject(new Error('Could not load grayscale keyframe.'));
    view.src = '/frame?state=pre&name=' + encodeURIComponent(f.name);
  });
  await loadMaskFromUrl('/api/mask_png?id=' + encodeURIComponent(reviewObj)
    + '&name=' + encodeURIComponent(f.name) + '&t=' + Date.now());
  grayReady = true; grayDirty = false; showMask(true);
  slider.value = idx; updateLabel(); renderMaskmap(); grayStatus();
  document.getElementById('clickStatus').textContent = f.excluded
    ? 'Scene-change-excluded frame: whole-image mask protected; editing is disabled.'
    : f.source_missing && !f.draft
    ? 'No mask for this keyframe. Paint a mask; Export includes it in the new folder.'
    : 'Brush / erase to correct · z saves draft · Export writes grayscale_mask_export/';
}
async function graySave(){
  if(!grayDirty) return;
  if(!grayReady) throw new Error('Wait for the keyframe and mask to load before saving.');
  const d = await grayJson('/api/save_edit', {
    id: reviewObj, frame: frames[idx].name, mask: maskCanvas.toDataURL('image/png')});
  frames[idx].px = d.px; frames[idx].has_mask = true; frames[idx].draft = true;
  grayPending = d.pending; grayDirty = false;
  document.getElementById('clickStatus').textContent = 'Draft saved. Export writes the corrected set; original masks stay unchanged.';
  grayStatus(); updateLabel();
}
saveEdit = () => grayRun(graySave);
cancelEdit = () => grayRun(grayLoadFrame);
startEdit = () => { if(!grayReady) grayRun(grayLoadFrame); };
showReview = () => grayRun(grayLoadFrame);
gotoIdx = i => grayRun(async () => {
  if(!frames.length) return;
  await graySave();
  idx = Math.max(0, Math.min(frames.length - 1, i));
  await grayLoadFrame();
});
exportSegments = () => grayRun(async () => {
  const status = document.getElementById('exportStatus');
  status.textContent = 'Exporting…';
  try {
    await graySave();
    const d = await grayJson('/api/export', {});
    grayPending = d.pending;
    frames.forEach(f => { f.source_missing = false; f.draft = false; });
    status.textContent = d.msg;
  } catch(e){ status.textContent = 'Export failed: ' + e.message; throw e; }
});

// This mode has exactly one object. No create/delete/group/propagation writers.
refreshObjects = async function(){
  const d = await grayJson('/api/objects');
  objsCache = d.objects; grayPending = d.pending;
  const card = document.createElement('div'); card.className = 'obj';
  card.textContent = `Invalid depth · ${Object.values(d.objects)[0].n_masks} keyframes · one object`;
  document.getElementById('objects').replaceChildren(card);
  grayStatus();
};
window.addEventListener('beforeunload', e => {
  if(grayDirty || grayBusy){ e.preventDefault(); e.returnValue = ''; }
});
// Suppress RGB-only shortcuts before the shared keyboard handler sees them.
document.addEventListener('keydown', e => {
  if(['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if(!e.ctrlKey && !e.metaKey && ['a', 'd', 'g', 'u', 'e', 'v', 'm', 'n'].includes(e.key.toLowerCase())){
    e.preventDefault(); e.stopImmediatePropagation();
  }
}, true);

grayRun(async () => {
  await refreshObjects();
  reviewObj = Object.keys(objsCache)[0]; reviewState = 'pre'; mode = 'review';
  const d = await grayJson('/api/object_frames?id=' + encodeURIComponent(reviewObj));
  frames = d.frames; idx = 0; setSliderMax(frames.length);
  document.getElementById('reviewBar').style.display = '';
  document.getElementById('reviewName').textContent = 'Invalid depth';
  document.getElementById('annotateHint').style.display = 'none';
  setTool('brush'); brushSize = 12; document.getElementById('brushSize').value = brushSize;
  await grayLoadFrame();
});
