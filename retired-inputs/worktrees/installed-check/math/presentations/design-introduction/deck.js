(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('frame-data').textContent);
  const frames = [...document.querySelectorAll('.frame')];
  const stage = document.getElementById('stage');
  const prev = document.getElementById('prev'), next = document.getElementById('next');
  const notesDialog = document.getElementById('notes-dialog');
  const tocDialog = document.getElementById('toc-dialog');
  let index = 0;
  function resize() {
    const view = document.getElementById('viewport');
    stage.style.transform = `scale(${Math.max(.1, Math.min((view.clientWidth - 24) / 1600, (view.clientHeight - 22) / 900))})`;
  }
  function show(value, hash = true) {
    index = Math.max(0, Math.min(frames.length - 1, value));
    frames.forEach((frame, n) => {
      frame.classList.toggle('active', n === index);
      frame.setAttribute('aria-hidden', String(n !== index));
    });
    const d = data[index];
    prev.disabled = index === 0; next.disabled = index === frames.length - 1;
    document.getElementById('position').textContent = `${d.slide + 1}.${d.step + 1} · ${index + 1}/${frames.length}`;
    document.getElementById('live').textContent = d.title;
    if (hash) history.replaceState(null, '', `#${d.slide+1}.${d.step+1}`);
  }
  function fromHash() {
    const match = location.hash.match(/^#(\d+)\.(\d+)$/);
    if (!match) return show(0, false);
    const target = data.findIndex(d => d.slide === Number(match[1])-1 && d.step === Number(match[2])-1);
    show(target < 0 ? 0 : target, false);
  }
  function notes() {
    const d = data[index];
    document.getElementById('notes-title').textContent = `${d.slide+1}.${d.step+1} · ${d.title}`;
    document.getElementById('notes-body').textContent = d.notes;
    const list = document.getElementById('notes-sources'); list.replaceChildren();
    for (const ref of d.refs) {
      const li = document.createElement('li'); const a = document.createElement('a');
      a.textContent = ref.label; a.href = ref.href; a.target = '_blank'; a.rel = 'noopener';
      li.append(a); list.append(li);
    }
    notesDialog.showModal();
  }
  const seen = new Set();
  data.forEach((d, n) => {
    if (seen.has(d.slide)) return; seen.add(d.slide);
    const b = document.createElement('button');
    b.dataset.stopId = d.id;
    b.textContent = `${String(d.slide+1).padStart(2,'0')}  ${d.title}`;
    b.addEventListener('click', () => { show(n); tocDialog.close(); });
    document.getElementById('toc-list').append(b);
  });
  prev.addEventListener('click', () => show(index-1));
  next.addEventListener('click', () => show(index+1));
  document.getElementById('replay').addEventListener('click', () => show(data.findIndex(d => d.slide === data[index].slide)));
  document.getElementById('notes').addEventListener('click', notes);
  document.getElementById('toc').addEventListener('click', () => tocDialog.showModal());
  document.getElementById('fullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch (_) { document.getElementById('live').textContent = '当前浏览器未进入全屏，可用浏览器全屏功能。'; }
  });
  document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => b.closest('dialog').close()));
  document.addEventListener('keydown', e => {
    if (notesDialog.open || tocDialog.open || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.matches('button, a, input, textarea, select') && [' ','Enter'].includes(e.key)) return;
    if (['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)) {e.preventDefault();show(index+1);}
    else if (['ArrowLeft','ArrowUp','PageUp'].includes(e.key)) {e.preventDefault();show(index-1);}
    else if (e.key === 'Home') {e.preventDefault();show(0);}
    else if (e.key === 'End') {e.preventDefault();show(frames.length-1);}
    else if (e.key.toLowerCase() === 'n') notes();
    else if (e.key.toLowerCase() === 'g') tocDialog.showModal();
  });
  let touchX=null;
  stage.addEventListener('touchstart', e => {touchX=e.touches.length===1?e.touches[0].clientX:null;},{passive:true});
  stage.addEventListener('touchend', e => {if(touchX!==null){const dx=e.changedTouches[0].clientX-touchX;if(Math.abs(dx)>60)show(index+(dx<0?1:-1));}touchX=null;},{passive:true});
  addEventListener('resize',resize);addEventListener('hashchange',fromHash);
  window.tideDeck = { data, show, get index(){return index;}, get total(){return frames.length;} };
  resize();fromHash();
})();
