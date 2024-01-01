var selected = 0, max_type = 0;

function closeControlTip(byclick){
  var tip = document.getElementById('control_tip');
  tip.style.display = "none";
}
function show(n) {
  if(n < 0 || n > max_type) return null;
  var examples = document.getElementById('examples');
  var controls = document.getElementById('controls');
  examples.className = 's' + n;
  controls.className = 'navbar-start s'+n;
}
function down() {
  if(selected == 0) return;
  show(1);
}
function up() {
  show(selected);
}
function over(n) {
  selected = n;
  show(n);
}
function code(e) {
  e = e || window.event;
  return(e.keyCode || e.which);
}

function keydown(e){
  let k = code(e);
  if(k >= 49 && k <= 57){
    return show(k - 48);
  }else if(k >= 97 && k <= 105){
    return show(k - 96);
  }

}

function set(selector, content){
  document.querySelector(selector).innerHTML = content;
}

function setCssVar(variable, value){
  document.documentElement.style.setProperty(variable, value);
}

function bodyListener(){
  document.body.addEventListener('mousedown', down);
  document.body.addEventListener('mouseup', up);
  document.body.addEventListener('mouseleave', up);
  document.body.addEventListener('keydown', keydown);
  //Bulma's Burger
}

function bulmaBurger(){
 // Get all "navbar-burger" elements
 const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
 // Check if there are any navbar burgers
 if ($navbarBurgers.length > 0) {
    // Add a click event on each of them
    $navbarBurgers.forEach( el => {
      el.addEventListener('click', () => {
        // Get the target from the "data-target" attribute
        const target = el.dataset.target;
        const $target = document.getElementById(target);
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');
      });
    });
  }
}

function applyZoombox(zoombox){
  
  if(zoombox.desktop){
    if(zoombox.desktop.size) setCssVar('--zoombox-size-desktop', zoombox.desktop.size);
    if(zoombox.desktop.scale) setCssVar('--zoombox-scale-desktop', zoombox.desktop.scale);
  }
  if(zoombox.tablet){
    if(zoombox.tablet.size) setCssVar('--zoombox-size-tablet', zoombox.tablet.size);
    if(zoombox.tablet.scale) setCssVar('--zoombox-scale-tablet', zoombox.tablet.scale);
  }
  if(zoombox.phone){
    console.log(zoombox.phone)
    if(zoombox.phone.size) setCssVar('--zoombox-size-phone', zoombox.phone.size);
    if(zoombox.phone.scale) setCssVar('--zoombox-scale-phone', zoombox.phone.scale);
  }
}

function main(){
  document.title = `Comparison on ${app.owner} ${app.name} dataset - NeX: Real-time View Synthesis with Neural Basis Expansion`;
  max_type = app.methods.length;
  set('body',getBody());
  bodyListener();
  bulmaBurger();
  set('#dataset-name',app.name);
  set('#dataset-owner',app.owner);
  set('#dataset-count-method', max_type);
  set('#controls', getControls(app.methods));
  set('#examples', getScenes(app.scenes, app.methods));
  setCssVar('--method-count', max_type);
  if(app.zoombox){
    applyZoombox(app.zoombox);
  }
}
    

document.addEventListener('DOMContentLoaded',main);
    

