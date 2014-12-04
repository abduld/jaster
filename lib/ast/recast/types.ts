/// <reference path="../types/def/core.ts" />
/// <reference path="../types/def/e4x.ts" />
/// <reference path="../types/def/es6.ts" />
/// <reference path="../types/def/es7.ts" />
/// <reference path="../types/def/fb-harmony.ts" />
/// <reference path="../types/def/mozilla.ts" />
module lib.ast.recast {
    export import types = lib.ast.types;
    import defs = types.definitions;
    var def = types.Type.def;
    def("File")
        .bases("Node")
        .build("program")
        .field("program", def("Program"));
    types.finalize();
}

