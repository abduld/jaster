
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


// Note that none of these types are buildable because the Mozilla Parser
// API doesn't specify any builder functions, and nobody uses E4X anymore.

        def("XMLDefaultDeclaration")
            .bases("Declaration")
            .field("namespace", def("Expression"));

        def("XMLAnyName").bases("Expression");

        def("XMLQualifiedIdentifier")
            .bases("Expression")
            .field("left", or(def("Identifier"), def("XMLAnyName")))
            .field("right", or(def("Identifier"), def("Expression")))
            .field("computed", isBoolean);

        def("XMLFunctionQualifiedIdentifier")
            .bases("Expression")
            .field("right", or(def("Identifier"), def("Expression")))
            .field("computed", isBoolean);

        def("XMLAttributeSelector")
            .bases("Expression")
            .field("attribute", def("Expression"));

        def("XMLFilterExpression")
            .bases("Expression")
            .field("left", def("Expression"))
            .field("right", def("Expression"));

        def("XMLElement")
            .bases("XML", "Expression")
            .field("contents", [def("XML")]);

        def("XMLList")
            .bases("XML", "Expression")
            .field("contents", [def("XML")]);

        def("XML").bases("Node");

        def("XMLEscape")
            .bases("XML")
            .field("expression", def("Expression"));

        def("XMLText")
            .bases("XML")
            .field("text", isString);

        def("XMLStartTag")
            .bases("XML")
            .field("contents", [def("XML")]);

        def("XMLEndTag")
            .bases("XML")
            .field("contents", [def("XML")]);

        def("XMLPointTag")
            .bases("XML")
            .field("contents", [def("XML")]);

        def("XMLName")
            .bases("XML")
            .field("contents", or(isString, [def("XML")]));

        def("XMLAttribute")
            .bases("XML")
            .field("value", isString);

        def("XMLCdata")
            .bases("XML")
            .field("contents", isString);

        def("XMLComment")
            .bases("XML")
            .field("contents", isString);

        def("XMLProcessingInstruction")
            .bases("XML")
            .field("target", isString)
            .field("contents", or(isString, null));
    }
}