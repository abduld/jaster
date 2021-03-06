/// <reference path="../../utils/utils.ts" />


module lib.ast {
    export module types {
        import Node = lib.ast.esprima.Syntax.Node;
        import assert = lib.utils.assert;

        var Ap: Array<any> = Array.prototype;
        var slice = Ap.slice;
        var map = Ap.map;
        var each = Ap.forEach;
        var Op = Object.prototype;
        var objToStr = Op.toString;
        var funObjStr = objToStr.call(function() {
        });
        var strObjStr = objToStr.call("");
        var hasOwn = Op.hasOwnProperty;

        // A type is an object with a .check method that takes a value and returns
        // true or false according to whether the value matches the type.

        export class Type {
            name: any;
            check: (val: any, deep?: any) => any;

            constructor(check, name) {
                var self = this;
                assert.ok(self instanceof Type);

                // Unfortunately we can't elegantly reuse isFunction and isString,
                // here, because this code is executed while defining those types.
                assert.strictEqual(objToStr.call(check), funObjStr,
                    check + " is not a function");

                // The `name` parameter can be either a function or a string.
                var nameObjStr = objToStr.call(name);
                assert.ok(nameObjStr === funObjStr ||
                    nameObjStr === strObjStr,
                    name + " is neither a function nor a string");
                this.name = name;
                this.check = function(value, deep?) {
                    var result = check.call(self, value, deep);
                    if (!result && deep && objToStr.call(deep) === funObjStr)
                        deep(self, value);
                    return result;
                }
            }

            // Like .check, except that failure triggers an AssertionError.
            assert(value, deep?) {
                if (!this.check(value, deep)) {
                    var str = shallowStringify(value);
                    assert.ok(false, str + " does not match type " + this);
                    return false;
                }
                return true;
            }


            toString() {
                var name = this.name;

                if (isString.check(name))
                    return name;

                if (isFunction.check(name))
                    return name.call(this) + "";

                return name + " type";
            }

            // Returns a type that matches the given value iff any of type1, type2,
            // etc. match the value.
            static or(...args: any[]) {
                var types = [];
                var len = arguments.length;
                for (var i = 0; i < len; ++i)
                    types.push(toType(args[i]));

                return new Type(function(value, deep) {
                    for (var i = 0; i < len; ++i)
                        if (types[i].check(value, deep))
                            return true;
                    return false;
                }, function() {
                        return types.join(" | ");
                    });
            }

            static fromArray(arr) {
                assert.ok(isArray.check(arr));
                assert.strictEqual(
                    arr.length, 1,
                    "only one element type is permitted for typed arrays");
                return toType(arr[0]).arrayOf();
            }

            arrayOf() {
                var elemType = this;
                return new Type(function(value, deep) {
                    return isArray.check(value) && value.every(function(elem) {
                        return elemType.check(elem, deep);
                    });
                }, function() {
                        return "[" + elemType + "]";
                    });
            }

            static fromObject(obj) {
                var fields: Field[] = Object.keys(obj).map(function(name) {
                    return new Field(name, obj[name]);
                });

                return new Type(function(value, deep) {
                    return isObject.check(value) && fields.every(function(field: Field) {
                        return field.type.check(value[field.name], deep);
                    });
                }, function() {
                        return "{ " + fields.join(", ") + " }";
                    });
            }

            // Define a type whose name is registered in a namespace (the defCache) so
            // that future definitions will return the same type given the same name.
            // In particular, this system allows for circular and forward definitions.
            // The Def object d returned from Type.def may be used to configure the
            // type d.type by calling methods such as d.bases, d.build, and d.field.
            static def(typeName): Def {
                isString.assert(typeName);
                return hasOwn.call(defCache, typeName)
                    ? defCache[typeName]
                    : defCache[typeName] = new Def(typeName);
            }
        }

        export var builtInTypes: { [name: string]: Type; } = {};

        function defBuiltInType(example, name): Type {
            var objStr = objToStr.call(example);

            Object.defineProperty(builtInTypes, name, {
                enumerable: true,
                value: new Type(function(value) {
                    return objToStr.call(value) === objStr;
                }, name)
            });

            return builtInTypes[name];
        }

        // These types check the underlying [[Class]] attribute of the given
        // value, rather than using the problematic typeof operator. Note however
        // that no subtyping is considered; so, for instance, isObject.check
        // returns false for [], /./, new Date, and null.
        var isString = defBuiltInType("", "string");
        var isFunction = defBuiltInType(function() {
        }, "function");
        var isArray = defBuiltInType([], "array");
        var isObject = defBuiltInType({}, "object");
        var isRegExp = defBuiltInType(/./, "RegExp");
        var isDate = defBuiltInType(new Date, "Date");
        var isNumber = defBuiltInType(3, "number");
        var isBoolean = defBuiltInType(true, "boolean");
        var isNull = defBuiltInType(null, "null");
        var isUndefined = defBuiltInType(void 0, "undefined");

        // There are a number of idiomatic ways of expressing types, so this
        // function serves to coerce them all to actual Type objects. Note that
        // providing the name argument is not necessary in most cases.
        function toType(from, name?) {
            // The toType function should of course be idempotent.
            if (from instanceof Type)
                return from;

            // The Def type is used as a helper for constructing compound
            // interface types for AST nodes.
            if (from instanceof Def)
                return from.type;

            // Support [ElemType] syntax.
            if (isArray.check(from))
                return Type.fromArray(from);

            // Support { someField: FieldType, ... } syntax.
            if (isObject.check(from))
                return Type.fromObject(from);

            // If isFunction.check(from), assume that from is a binary predicate
            // function we can use to define the type.
            if (isFunction.check(from))
                return new Type(from, name);

            // As a last resort, toType returns a type that matches any value that
            // is === from. This is primarily useful for literal values like
            // toType(null), but it has the additional advantage of allowing
            // toType to be a total function.
            return new Type(function(value) {
                return value === from;
            }, isUndefined.check(name) ? function() {
                    return from + "";
                } : name);
        }


        class Field {
            name: any;
            type: Type;
            hidden: any;
            defaultFn: any;

            constructor(name, type, defaultFn?, hidden?) {
                var self = this;

                assert.ok(self instanceof Field);
                isString.assert(name);

                type = toType(type);


                if (isFunction.check(defaultFn)) {
                    this.defaultFn = defaultFn;
                }
                this.name = name;
                this.type = type;
                this.hidden = !!hidden;
            }

            toString() {
                return JSON.stringify(this.name) + ": " + this.type;
            }

            getValue(obj) {
                var value = this.name;

                if (!isUndefined.check(value))
                    return value;

                if (this.defaultFn && this.defaultFn.value)
                    value = this.defaultFn.value.call(obj);

                return value;
            }
        }


        function shallowStringify(value) {
            if (isObject.check(value))
                return "{" + Object.keys(value).map(function(key) {
                    return key + ": " + value[key];
                }).join(", ") + "}";

            if (isArray.check(value))
                return "[" + value.map(shallowStringify).join(", ") + "]";

            return JSON.stringify(value);
        }

        // In order to return the same Def instance every time Type.def is called
        // with a particular name, those instances need to be stored in a cache.
        var defCache = Object.create(null);

        export class Def {
            typeNames: any;
            baseNames: any[];
            ownFields: any;
            allSupertypes: any[];
            supertypeList: any[];
            fieldNames: any[];
            type: Type;
            finalized: boolean = false;
            typeName: any;
            allFields: any[];
            buildable: boolean = false;

            constructor(typeName) {
                var self = this;
                assert.ok(self instanceof Def);

                Object.defineProperties(self, {
                    typeName: { value: typeName },
                    baseNames: { value: [] },
                    ownFields: { value: Object.create(null) },

                    // These two are populated during finalization.
                    allSupertypes: { value: Object.create(null) }, // Includes own typeName.
                    supertypeList: { value: [] }, // Linear inheritance hierarchy.
                    allFields: { value: Object.create(null) }, // Includes inherited fields.
                    fieldNames: { value: [] }, // Non-hidden keys of allFields.

                    type: {
                        value: new Type(function(value, deep) {
                            return self.check(value, deep);
                        }, typeName)
                    }
                });
            }

            static fromValue(value) {
                if (value && typeof value === "object") {
                    var type = value.type;
                    if (typeof type === "string" &&
                        hasOwn.call(defCache, type)) {
                        var d = defCache[type];
                        if (d.finalized) {
                            return d;
                        }
                    }
                }

                return null;
            }

            isSupertypeOf(that) {
                if (that instanceof Def) {
                    assert.strictEqual(this.finalized, true);
                    assert.strictEqual(that.finalized, true);
                    return hasOwn.call(that.allSupertypes, this.typeName);
                } else {
                    assert.ok(false, that + " is not a Def");
                }
            }

            checkAllFields(value, deep) {
                var allFields = this.allFields;
                assert.strictEqual(this.finalized, true);

                function checkFieldByName(name) {
                    var field = allFields[name];
                    var type = field.type;
                    var child = field.getValue(value);
                    return type.check(child, deep);
                }

                return isObject.check(value)
                    && Object.keys(allFields).every(checkFieldByName);
            }

            check(value, deep?) {
                assert.strictEqual(
                    this.finalized, true,
                    "prematurely checking unfinalized type " + this.typeName);

                // A Def type can only match an object value.
                if (!isObject.check(value))
                    return false;

                var vDef = Def.fromValue(value);
                if (!vDef) {
                    // If we couldn't infer the Def associated with the given value,
                    // and we expected it to be a SourceLocation or a Position, it was
                    // probably just missing a "type" field (because Esprima does not
                    // assign a type property to such nodes). Be optimistic and let
                    // this.checkAllFields make the final decision.
                    if (this.typeName === "SourceLocation" ||
                        this.typeName === "Position") {
                        return this.checkAllFields(value, deep);
                    }

                    // Calling this.checkAllFields for any other type of node is both
                    // bad for performance and way too forgiving.
                    return false;
                }

                // If checking deeply and vDef === this, then we only need to call
                // checkAllFields once. Calling checkAllFields is too strict when deep
                // is false, because then we only care about this.isSupertypeOf(vDef).
                if (deep && vDef === this)
                    return this.checkAllFields(value, deep);

                // In most cases we rely exclusively on isSupertypeOf to make O(1)
                // subtyping determinations. This suffices in most situations outside
                // of unit tests, since interface conformance is checked whenever new
                // instances are created using builder functions.
                if (!this.isSupertypeOf(vDef))
                    return false;

                // The exception is when deep is true; then, we recursively check all
                // fields.
                if (!deep)
                    return true;

                // Use the more specific Def (vDef) to perform the deep check, but
                // shallow-check fields defined by the less specific Def (this).
                return vDef.checkAllFields(value, deep)
                    && this.checkAllFields(value, false);
            }

            bases(...args: any[]): Def {
                var bases = this.baseNames;

                assert.strictEqual(this.finalized, false);

                each.call(args, function(baseName) {
                    isString.assert(baseName);

                    // This indexOf lookup may be O(n), but the typical number of base
                    // names is very small, and indexOf is a native Array method.
                    if (bases.indexOf(baseName) < 0)
                        bases.push(baseName);
                });

                return this; // For chaining.
            }


            finalize() {
                // It's not an error to finalize a type more than once, but only the
                // first call to .finalize does anything.
                if (!this.finalized) {
                    var allFields = this.allFields;
                    var allSupertypes = this.allSupertypes;

                    this.baseNames.forEach(function(name) {
                        var def = defCache[name];
                        if (lib.utils.isUndefined(def)) {
                            return;
                        }
                        def.finalize();
                        extend(allFields, def.allFields);
                        extend(allSupertypes, def.allSupertypes);
                    });

                    // TODO Warn if fields are overridden with incompatible types.
                    extend(allFields, this.ownFields);
                    allSupertypes[this.typeName] = this;

                    this.fieldNames.length = 0;
                    for (var fieldName in allFields) {
                        if (hasOwn.call(allFields, fieldName) && !allFields[fieldName].hidden) {
                            this.fieldNames.push(fieldName);
                        }
                    }

                    // Types are exported only once they have been finalized.
                    Object.defineProperty(namedTypes, this.typeName, {
                        enumerable: true,
                        value: this.type
                    });

                    Object.defineProperty(this, "finalized", { value: true });

                    // A linearization of the inheritance hierarchy.
                    populateSupertypeList(this.typeName, this.supertypeList);
                }
            }

            buildParams: any;
            // Calling the .build method of a Def simultaneously marks the type as
            // buildable (by defining builders[getBuilderName(typeName)]) and
            // specifies the order of arguments that should be passed to the builder
            // function to create an instance of the type.
            build(...args: any[]) {
                var self = this;

                // Calling Def.prototype.build multiple times has the effect of merely
                // redefining this property.
                Object.defineProperty(self, "buildParams", {
                    value: slice.call(args),
                    writable: false,
                    enumerable: false,
                    configurable: true
                });

                assert.strictEqual(self.finalized, false);
                isString.arrayOf().assert(self.buildParams);

                if (self.buildable) {
                    // If this Def is already buildable, update self.buildParams and
                    // continue using the old builder function.
                    return self;
                }

                // Every buildable type will have its "type" field filled in
                // automatically. This includes types that are not subtypes of Node,
                // like SourceLocation, but that seems harmless (TODO?).
                self.field("type", self.typeName, function() {
                    return self.typeName
                });

                // Override Dp.buildable for this Def instance.
                Object.defineProperty(self, "buildable", { value: true });

                Object.defineProperty(builders, getBuilderName(self.typeName), {
                    enumerable: true,

                    value: function(...args: any[]) {
                        var argc = args.length;
                        var built = Object.create(nodePrototype);

                        assert.ok(
                            self.finalized,
                            "attempting to instantiate unfinalized type " + self.typeName);
                        function add(param, i?) {
                            if (hasOwn.call(built, param))
                                return;

                            var all = self.allFields;
                            assert.ok(hasOwn.call(all, param), param);

                            var field: Field = all[param];

                            var type = field.type;
                            var value;
                            if (isNumber.check(i) && i < argc) {
                                value = args[i];
                            } else if (!lib.utils.isUndefined(field.defaultFn)) {
                                // Expose the partially-built object to the default
                                // function as its `this` object.
                                value = field.defaultFn.call(built, args);
                            } else {
                                var message = "no value or default function given for field " +
                                    JSON.stringify(param) + " of " + self.typeName + "(" +
                                    self.buildParams.map(function(name) {
                                        return all[name];
                                    }).join(", ") + ")";
                                assert.ok(false, message);
                            }

                            if (!type.check(value)) {
                                assert.ok(
                                    false,
                                    shallowStringify(value) +
                                    " does not match field " + field +
                                    " of type " + self.typeName
                                    );
                            }

                            // TODO Could attach getters and setters here to enforce
                            // dynamic type safety.
                            built[param] = value;
                        }

                        self.buildParams.forEach(function(param, i) {
                            add(param, i);
                        });

                        Object.keys(self.allFields).forEach(function(param) {
                            add(param); // Use the default value.
                        });

                        // Make sure that the "type" field was filled automatically.
                        assert.strictEqual(built.type, self.typeName);

                        return built;
                    }
                });

                return self; // For chaining.
            }

            // The reason fields are specified using .field(...) instead of an object
            // literal syntax is somewhat subtle: the object literal syntax would
            // support only one key and one value, but with .field(...) we can pass
            // any number of arguments to specify the field.
            field(name, type, defaultFn?, hidden?): Def {
                assert.strictEqual(this.finalized, false);
                this.ownFields[name] = new Field(name, type, defaultFn, hidden);
                return this; // For chaining.
            }
        }


        // Note that the list returned by this function is a copy of the internal
        // supertypeList, *without* the typeName itself as the first element.
        export function getSupertypeNames(typeName) {
            assert.ok(hasOwn.call(defCache, typeName));
            var d = defCache[typeName];
            assert.strictEqual(d.finalized, true);
            return d.supertypeList.slice(1);
        };

        // Returns an object mapping from every known type in the defCache to the
        // most specific supertype whose name is an own property of the candidates
        // object.
        export function computeSupertypeLookupTable(candidates) {
            var table = {};
            var typeNames = Object.keys(defCache);
            var typeNameCount = typeNames.length;

            for (var i = 0; i < typeNameCount; ++i) {
                var typeName = typeNames[i];
                var d = defCache[typeName];
                assert.strictEqual(d.finalized, true);
                for (var j = 0; j < d.supertypeList.length; ++j) {
                    var superTypeName = d.supertypeList[j];
                    if (hasOwn.call(candidates, superTypeName)) {
                        table[typeName] = superTypeName;
                        break;
                    }
                }
            }

            return table;
        };


        export var builders: { [name: string]: (...args: any[]) => any; } = {};

        // This object is used as prototype for any node created by a builder.
        var nodePrototype = {};

        // Call this function to define a new method to be shared by all AST
        // nodes. The replaced method (if any) is returned for easy wrapping.
        export function defineMethod(name, func) {
            var old = nodePrototype[name];

            // Pass undefined as func to delete nodePrototype[name].
            if (isUndefined.check(func)) {
                delete nodePrototype[name];

            } else {
                isFunction.assert(func);

                Object.defineProperty(nodePrototype, name, {
                    enumerable: true, // For discoverability.
                    configurable: true, // For delete proto[name].
                    value: func
                });
            }

            return old;
        };


        function getBuilderName(typeName) {
            return typeName.replace(/^[A-Z]+/, function(upperCasePrefix) {
                var len = upperCasePrefix.length;
                switch (len) {
                    case 0:
                        return "";
                    // If there's only one initial capital letter, just lower-case it.
                    case 1:
                        return upperCasePrefix.toLowerCase();
                    default:
                        // If there's more than one initial capital letter, lower-case
                        // all but the last one, so that XMLDefaultDeclaration (for
                        // example) becomes xmlDefaultDeclaration.
                        return upperCasePrefix.slice(
                            0, len - 1).toLowerCase() +
                            upperCasePrefix.charAt(len - 1);
                }
            });
        }


        export var namedTypes: { [name: string]: Type; } = {};

        // Like Object.keys, but aware of what fields each AST type should have.
        export function getFieldNames(object) {
            var d = Def.fromValue(object);
            if (d) {
                return d.fieldNames.slice(0);
            }

            if ("type" in object) {
                assert.ok(
                    false,
                    "did not recognize object of type " +
                    JSON.stringify(object.type)
                    );
            }

            return Object.keys(object);
        }

        // Get the value of an object property, taking object.type and default
        // functions into account.
        export function getFieldValue(object, fieldName) {
            var d = Def.fromValue(object);
            if (d) {
                var field = d.allFields[fieldName];
                if (field) {
                    return field.getValue(object);
                }
            }

            return object[fieldName];
        }

        // Iterate over all defined fields of an object, including those missing
        // or undefined, passing each field name and effective value (as returned
        // by getFieldValue) to the callback. If the object has no corresponding
        // Def, the callback will never be called.
        export function eachField(object, callback, context?) {
            getFieldNames(object).forEach(function(name) {
                callback.call(this, name, getFieldValue(object, name));
            }, context);
        };

        // Similar to eachField, except that iteration stops as soon as the
        // callback returns a truthy value. Like Array.prototype.some, the final
        // result is either true or false to indicates whether the callback
        // returned true for any element or not.
        export function someField(object, callback, context?) {
            return getFieldNames(object).some(function(name) {
                return callback.call(this, name, getFieldValue(object, name));
            }, context);
        };


        function populateSupertypeList(typeName, list) {
            list.length = 0;
            list.push(typeName);

            var lastSeen = Object.create(null);

            for (var pos = 0; pos < list.length; ++pos) {
                typeName = list[pos];
                var d = defCache[typeName];
                assert.strictEqual(d.finalized, true);

                // If we saw typeName earlier in the breadth-first traversal,
                // delete the last-seen occurrence.
                if (hasOwn.call(lastSeen, typeName)) {
                    delete list[lastSeen[typeName]];
                }

                // Record the new index of the last-seen occurrence of typeName.
                lastSeen[typeName] = pos;

                // Enqueue the base names of this type.
                list.push.apply(list, d.baseNames);
            }

            // Compaction loop to remove array holes.
            for (var to = 0, from = to, len = list.length; from < len; ++from) {
                if (hasOwn.call(list, from)) {
                    list[to++] = list[from];
                }
            }

            list.length = to;
        }

        function extend(into, from) {
            Object.keys(from).forEach(function(name) {
                into[name] = from[name];
            });

            return into;
        };

        export function finalize() {
            Object.keys(defCache).forEach(function(name) {
                var n: Def = defCache[name];
                if (!lib.utils.isUndefined(n)) {
                    n.finalize();
                }
            });
        };

        export module shared {
            var builtin = lib.ast.types.builtInTypes;
            var isNumber: Type = builtin["number"];

            // An example of constructing a new type with arbitrary constraints from
            // an existing type.
            export function geq(than) {
                return new Type(function(value) {
                    return isNumber.check(value) && value >= than;
                }, isNumber + " >= " + than);
            };

            // Default value-returning functions that may optionally be passed as a
            // third argument to Def.prototype.field.
            export var defaults = {
                // Functions were used because (among other reasons) that's the most
                // elegant way to allow for the emptyArray one always to give a new
                // array instance.
                "null": function() {
                    return null
                },
                "emptyArray": function() {
                    return []
                },
                "false": function() {
                    return false
                },
                "true": function() {
                    return true
                },
                "undefined": function() {
                },
                "identity": function(id) {
                    return id;
                },
                "location": function(args) {
                    return _.isUndefined(args) ? null :
                        _.findWhere(args, (arg: any) => _.isObject(arg) && arg.type === "SourceLocation") || null;
                }
            };

            var naiveIsPrimitive = Type.or(
                builtin["string"],
                builtin["number"],
                builtin["boolean"],
                builtin["null"],
                builtin["undefined"]
                );

            export var isPrimitive = new Type(function(value) {
                if (value === null)
                    return true;
                var type = typeof value;
                return !(type === "object" ||
                    type === "function");
            }, naiveIsPrimitive.toString());
        }

        export import geq = shared.geq;
        export import isPrimitive = shared.isPrimitive;
        export import defaults = shared.defaults;
    }
}
