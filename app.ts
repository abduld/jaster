
/// <reference path="./typings/assert/assert.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="./typings/parallel/parallel.d.ts" />
/// <reference path="./typings/q/Q.d.ts" />
/// <reference path="./typings/jquery/jquery.d.ts" />
/// <reference path="./typings/codemirror/codemirror.d.ts" />
/// <reference path="./typings/requirejs/require.d.ts" />
/// <reference path="./lib/ref.ts" />



var code: string;


var vizNode : any = undefined;
function visualize(options? : number) {
  var dom = $("#visualization")[0];
  if (_.isUndefined(options)) {
    options = lib.utils.castTo<any>($(".ccode")[0]).value.indexOf("<=") > 0 ? 1 : 0;
}
  $(".visualize-btn").hide();
  var rect = lib.viz.gridVisualization({
    blockDim: new lib.cuda.Dim3(4, 4),
    gridDim: new lib.cuda.Dim3(8, 8),
    options: options
    });
    console.log(vizNode);
    vizNode = React.render(rect, dom, () =>$(".visualize-btn").show() );
  }

function initWebApp() {

  //visualize();
    /*
      var cudaEditor = CodeMirror.fromTextArea(lib.utils.castTo<HTMLTextAreaElement>($("#cuda-code")[0]), {
          lineNumbers: true,
          readOnly: true,
          mode: "text/x-c++src"
      });
      var cudaDoc = cudaEditor.getDoc();
      cudaDoc.setValue(lib.example.mp2Source);
      cudaEditor.setSize("100%", 1000);
    var ast = lib.ast.importer.cena.fromCena(lib.example.mp1);
    var res = lib.ast.gen.generate(
        ast.toEsprima(),
        // we might have to do some extra think here (see https://github.com/estools/escodegen/wiki/Source-Map-Usage )
        { sourceMap: true, sourceMapWithCode: true, comment: true, indent: true, sourceContent: ast.cform }
        );

    var jsEditor = CodeMirror.fromTextArea(lib.utils.castTo<HTMLTextAreaElement>($("#js-code")[0]), {
        lineNumbers: true,
        readOnly: true,
        matchBrackets: true,
        mode: "javascript",
        theme: 'eclipse'
    });
    var jsDoc = jsEditor.getDoc();

    code = res.code;
    //code = code + "\n" + lib.utils.sourceMapToComment(res.map.toJSON());
    jsDoc.setValue(code);
    jsEditor.setSize("100%", 1000);

    code = res.code;

    global$.code = res;

    */

}

function initWorkerApp(event) {
    importScripts("Scripts/lodash.js");
    lib.parallel.actOnEvent(event);
}

if (lib.utils.ENVIRONMENT_IS_WEB) {
    $(initWebApp);
} else if (lib.utils.ENVIRONMENT_IS_WORKER) {
    //self.onmessage = initWorkerApp;
}
