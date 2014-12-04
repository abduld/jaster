module lib.ast.types {
    export module definitions {
        import core = lib.ast.types.definitions.core;
        import types = lib.ast.types;
        var def = types.Type.def;
        var or = types.Type.or;
        var geq = types.geq;

        def("ForOfStatement")
            .bases("Statement")
            .build("left", "right", "body")
            .field("left", or(
                def("VariableDeclaration"),
                def("Expression")))
            .field("right", def("Expression"))
            .field("body", def("Statement"));

        def("LetStatement")
            .bases("Statement")
            .build("head", "body")
            // TODO Deviating from the spec by reusing VariableDeclarator here.
            .field("head", [def("VariableDeclarator")])
            .field("body", def("Statement"));

        def("LetExpression")
            .bases("Expression")
            .build("head", "body")
            // TODO Deviating from the spec by reusing VariableDeclarator here.
            .field("head", [def("VariableDeclarator")])
            .field("body", def("Expression"));

        def("GraphExpression")
            .bases("Expression")
            .build("index", "expression")
            .field("index", geq(0))
            .field("expression", def("Literal"));

        def("GraphIndexExpression")
            .bases("Expression")
            .build("index")
            .field("index", geq(0));

        //       types.finalize();
    }
}