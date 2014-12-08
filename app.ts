
/// <reference path="./typings/assert/assert.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="./typings/parallel/parallel.d.ts" />
/// <reference path="./typings/q/Q.d.ts" />
/// <reference path="./typings/jquery/jquery.d.ts" />
/// <reference path="./typings/codemirror/codemirror.d.ts" />
/// <reference path="./typings/requirejs/require.d.ts" />
/// <reference path="./lib/ref.ts" />

$(() => {
    var cudaEditor = CodeMirror.fromTextArea(lib.utils.castTo<HTMLTextAreaElement>($("#cuda-code")[0]), {
        lineNumbers: true,
        readOnly: true,
        mode: "text/x-c++src"
    });
    var cudaDoc = cudaEditor.getDoc();
    cudaDoc.setValue(lib.example.mp2Source);
    cudaEditor.setSize("100%", 1000);

    var ast = lib.ast.importer.cena.fromCena(lib.example.mp2);
    var res = lib.ast.gen.generate(
        ast.toEsprima(),
        // we might have to do some extra think here (see https://github.com/estools/escodegen/wiki/Source-Map-Usage )
        { sourceMap: true, sourceMapWithCode: true, comment: true, indent: true , sourceContent: ast.cform }
    );


    var jsEditor = CodeMirror.fromTextArea(lib.utils.castTo<HTMLTextAreaElement>($("#js-code")[0]), {
        lineNumbers: true,
        readOnly: true,
        mode: "javascript"
    });
    var jsDoc = jsEditor.getDoc();
    jsDoc.setValue(res.code);
    jsEditor.setSize("100%", 800);
})
