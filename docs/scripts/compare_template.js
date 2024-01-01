// template defination for app

function getBody(){
  return `
  <nav class="navbar is-fixed-top is-light" role="navigation" aria-label="main navigation">
    <div class="navbar-brand">
      <p class="navbar-item">
        <a href="../" style="color:#4a4a4a"><b>NeX</b></a>
      </p>
      <p class="navbar-item">
            <a href="../" style="color:#4a4a4a">Go Back</a>
          </p>
      <a role="button" class="navbar-burger burger" aria-label="menu" aria-expanded="false" data-target="navbarBasicExample">
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
      </a>
     
    </div>
    <div id="navbarBasicExample" class="navbar-menu">
        <div id="controls" class="navbar-start s0"><!-- hover bar --></div>
    </div>    
  </nav>
  <section class="hero is-dark">
    <div class="hero-body">
      <div class="container is-fluid">
        <h1 class="title" >
          Comparison on <span id='dataset-owner'></span> <span id='dataset-name'></span> dataset
        </h1>
      </div>
    </div>
  </section>
  <section>
    <div id="control_tip" class="notification is-primary is-light"	>
      Press number 1 - <span id="dataset-count-method"></span> or hover over the top menu to switch between the methods
    </div>  
  </section>
  <section class="section">
    <div class="container is-fluid">
      <div id="examples" class="s0"><!-- scene show here --></div>
    </div>
  </section>
  `;
}

function getControlItem(name, id){
  return `
  <a class="navbar-item" onmouseenter="over(${id})">
    ${ id > 0 ? id + ') ' : ''}${name}
  </a>
  `;
}

function getControls(methods){
  var output = getControlItem('ALL',0);
  methods.forEach((name,id) => {
    output += getControlItem(name,id+1)
  });
  return output; 
}

function getScene(scene, methods){
  return `
    <div class="scene">
      ${scene.images.map((img,id) =>
        `
        <div class="scene-method">
          <img src="${scene.path}/${img}">
          <div class="zoombox" style=" background-image:url(${scene.path}/${img}); background-size: calc(${scene.width}px * var(--zoombox-scale) * ${scene.zoom.ratio}) calc(${scene.height}px *  var(--zoombox-scale) * ${scene.zoom.ratio}); background-position: calc(-${scene.zoom.left}px *  var(--zoombox-scale) * ${scene.zoom.ratio}) calc(-${scene.zoom.top}px *  var(--zoombox-scale)  * ${scene.zoom.ratio}) "></div>
          <span class="method-label label">${methods[id]}</span>
        </div>
        `
      ).join('\n')}
      <span class="scene-label label">${scene.name}</span>
    </div>
    <div style="display: block;"></div>
  `;
}


function getScenes(scenes, methods){
  return scenes.map((e)=>getScene(e,methods)).join('\n');
}