
/// <reference path="ref.ts" />

declare var requirejs;
requirejs.config({
    baseUrl: "dist",
    enforceDefine: true,
    paths: {
        react : "http://cdnjs.cloudflare.com/ajax/libs/react/0.12.0/react",
        jquery: "//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min",
        underscore: "http://underscorejs.org/underscore-min",
        bootstrap: "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/js/bootstrap.min",
        "react/addons": "http://fb.me/react-with-addons-0.12.1",
        index: "index"
    }
});
requirejs(["index"], function(ctx) {
    ctx.main();
});