/// <reference path="../types/def/core.ts" />
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

