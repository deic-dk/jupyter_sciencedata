// leave at least 2 line with only a star on it below, or doc generation fails
/**
 *
 *
 * Placeholder for custom user javascript
 * mainly to be overridden in profile/static/custom/custom.js
 * This will always be an empty file in IPython
 *
 * User could add any javascript in the `profile/static/custom/custom.js` file.
 * It will be executed by the ipython notebook at load time.
 *
 * Same thing with `profile/static/custom/custom.css` to inject custom css into the notebook.
 *
 *
 * The object available at load time depend on the version of IPython in use.
 * there is no guaranties of API stability.
 *
 * The example below explain the principle, and might not be valid.
 *
 * Instances are created after the loading of this file and might need to be accessed using events:
 *     define([
 *        'base/js/namespace',
 *        'base/js/promises'
 *     ], function(IPython, promises) {
 *         promises.app_initialized.then(function (appName) {
 *             if (appName !== 'NotebookApp') return;
 *             IPython.keyboard_manager....
 *         });
 *     });
 *
 * __Example 1:__
 *
 * Create a custom button in toolbar that execute `%qtconsole` in kernel
 * and hence open a qtconsole attached to the same kernel as the current notebook
 *
 *    define([
 *        'base/js/namespace',
 *        'base/js/promises'
 *    ], function(IPython, promises) {
 *        promises.app_initialized.then(function (appName) {
 *            if (appName !== 'NotebookApp') return;
 *            IPython.toolbar.add_buttons_group([
 *                {
 *                    'label'   : 'run qtconsole',
 *                    'icon'    : 'icon-terminal', // select your icon from http://fortawesome.github.io/Font-Awesome/icons
 *                    'callback': function () {
 *                        IPython.notebook.kernel.execute('%qtconsole')
 *                    }
 *                }
 *                // add more button here if needed.
 *                ]);
 *        });
 *    });
 *
 * __Example 2:__
 *
 * At the completion of the dashboard loading, load an unofficial javascript extension
 * that is installed in profile/static/custom/
 *
 *    define([
 *        'base/js/events'
 *    ], function(events) {
 *        events.on('app_initialized.DashboardApp', function(){
 *            requirejs(['custom/unofficial_extension.js'])
 *        });
 *    });
 *
 *
 *
 * @module IPython
 * @namespace IPython
 * @class customjs
 * @static
 */

(function() {
  // Append loading div to body
  var loaderDiv = document.createElement('div');
  var loadingDiv = document.createElement('div');
  loaderDiv.classList.add('loader');
  loadingDiv.classList.add('loading');
  loaderDiv.appendChild(loadingDiv);
  document.body.appendChild(loaderDiv);
})();

(function() {
  // Load the script
  const script = document.createElement("script");
  script.src = 'https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js';
  script.type = 'text/javascript';
  script.addEventListener('load', () => {
    console.log(`jQuery ${$.fn.jquery} has been loaded successfully!`);
    // use jQuery below
    function recursive() {
      console.log("Recursing. "+$('#notebook_list .list_item').length);
      setTimeout(function(){
       if(!$('#notebook_list').length || $('#notebook_list .list_item').length){
         $('#site').css('visibility', 'visible');
         $('.loader').hide();
       }
       else{
        recursive();
       }
      }, 100)
    }
    recursive();
  });
  document.head.appendChild(script);
  // Don't show Jupyter 7 migration message
  localStorage.setItem('showNbClassicNews', false);
  newsId.style.display = 'none';
})();

(function(send) {
  XMLHttpRequest.prototype.send = function(body) {
    this.addEventListener('readystatechange', function() {
      if (/*this.responseURL.includes('kube') && */this.readyState === 4) {
        // For some readon the autosave dropdown setting is cleared on saving.
        // This is an ugly hack to deal with this.
        if(!window.autosave_interval){
         window.autosave_interval = 0;
        }
        //$('select.ui-widget-content').val(window.autosave_interval);
        if($('select.ui-widget-content').val()!==window.autosave_interval && Jupyter.notebook){
          Jupyter.notebook.set_autosave_interval(window.autosave_interval);
        }
        $('.loader').hide();
      }
    }, false);
    //var info="send data\r\n"+body;
    $('a[href="#nbextensions_configurator"]').addClass('nbextensions_configurator_tab_link');
    //$('a[href="#nbextensions_configurator"]').hide();
   // $('a[href="#clusters"]').hide();
    window.autosave_interval = parseInt($('select.ui-widget-content').val());
    $('.loader').show();
    send.call(this, body);
};
})(XMLHttpRequest.prototype.send);

// This should no longer be necessary - patched in image
/*
(function() {
// Disable autosave
define([
  'base/js/namespace',
  'base/js/events'
  ],
  function(IPython, events) {
    events.on("notebook_loaded.Notebook",
      function () {
        Jupyter.notebook.set_autosave_interval(0);
        //in milliseconds
        $("select.ui-widget-content").val(0);
        $("select.ui-widget-content").find('option[value="2"]').text('2');
      }
    );
    //may include additional events.on() statements
  }
);
})();
*/
