module lib.ast.recast {
    export import types = lib.ast.types;
    var def = types.Type.def;

    def("File")
        .bases("Node")
        .build("program")
        .field("program", def("Program"));

    types.finalize();
}

