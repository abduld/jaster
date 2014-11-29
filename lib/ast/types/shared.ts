module lib.ast.types {

    var builtin = types.builtInTypes;
    var isNumber:Type = builtin["number"];

    export module shared {
// An example of constructing a new type with arbitrary constraints from
// an existing type.
        export function geq(than) {
            return new Type(function (value) {
                return isNumber.check(value) && value >= than;
            }, isNumber + " >= " + than);
        };

// Default value-returning functions that may optionally be passed as a
// third argument to Def.prototype.field.
        export var defaults = {
            // Functions were used because (among other reasons) that's the most
            // elegant way to allow for the emptyArray one always to give a new
            // array instance.
            "null": function () {
                return null
            },
            "emptyArray": function () {
                return []
            },
            "false": function () {
                return false
            },
            "true": function () {
                return true
            },
            "undefined": function () {
            }
        };

        var naiveIsPrimitive = Type.or(
            builtin["string"],
            builtin["number"],
            builtin["boolean"],
            builtin["null"],
            builtin["undefined"]
        );

        export var isPrimitive = new Type(function (value) {
            if (value === null)
                return true;
            var type = typeof value;
            return !(type === "object" ||
            type === "function");
        }, naiveIsPrimitive.toString());
    }
}