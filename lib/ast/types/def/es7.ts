

module lib.ast.types {
    export module definitions {
        import core = lib.ast.types.definitions.core;
        import types = lib.ast.types;
        var def = types.Type.def;
        var or = types.Type.or;
        var builtin = types.builtInTypes;
        var isString = builtin["string"];
        var isBoolean = builtin["boolean"];
        var defaults = types.defaults;
        var geq = types.geq;

        def("Function")
            .field("async", isBoolean, defaults["false"]);

        def("SpreadProperty")
            .bases("Node")
            .build("argument")
            .field("argument", def("Expression"));

        def("ObjectExpression")
            .field("properties", [or(def("Property"), def("SpreadProperty"))]);

        def("SpreadPropertyPattern")
            .bases("Pattern")
            .build("argument")
            .field("argument", def("Pattern"));

        def("ObjectPattern")
            .field("properties", [or(
                def("PropertyPattern"),
                def("SpreadPropertyPattern")
            )]);

        def("AwaitExpression")
            .bases("Expression")
            .build("argument", "all")
            .field("argument", or(def("Expression"), null))
            .field("all", isBoolean, defaults["false"]);
    }
}