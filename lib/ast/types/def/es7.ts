
/// <referench path="core.ts" />

module lib.ast.types {
    export module def {
        import core = lib.ast.types.def.core;
        import types = lib.ast.types;
        import shared = lib.ast.types.shared;
        var def = types.Type.def;
        var or = types.Type.or;
        var builtin = types.builtInTypes;
        var isString = builtin["string"];
        var isBoolean = builtin["boolean"];
        var defaults = shared.defaults;
        var geq = shared.geq;

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