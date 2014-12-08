/// <referench path="../../../typings/tsd.d.ts" />

module lib.ast {
    export module importer {
        export module cena {
            import esprima = lib.ast.esprima;
            import castTo = lib.utils.castTo;
            import builder_ = lib.ast.types.builders;
            import isUndefined = lib.utils.isUndefined;

            var builder:any = castTo<any>(builder_);

            function startsWith(s:string, str:string):boolean {
                return s.indexOf(str) === 0;
            };
            function unknownLocation():esprima.Syntax.LineLocation {
                return builder.sourceLocation(
                    builder.position(1, 1),
                    builder.position(1, 1),
                    "cuda.js"
                )
            };

            export class Node {
                type:string
                rloc:any
                loc:esprima.Syntax.LineLocation
                raw:string
                cform:string
                marker:any
                parent:Node
                deleted:boolean

                constructor(type:string, loc:any, raw:string, cform:string) {
                    this.type = type;
                    this.rloc = loc;
                    if (isUndefined(loc)) {
                        this.loc = unknownLocation();
                    } else {
                        this.loc = builder.sourceLocation(
                            builder.position(loc.start.line + 1, loc.start.column + 1),
                            builder.position(loc.end.line + 1, loc.end.column + 1),
                            "cuda.js"
                        );
                        ;
                    }
                    this.raw = raw;
                    this.cform = cform;
                    this.marker = {};
                    this.parent = null;
                    this.deleted = false;
                }

                static fromCena(o:any):Node {
                    return new Node("Unknown", unknownLocation(), "", "");
                }

                toEsprima():esprima.Syntax.Node {
                    if (this.deleted) {
                        return (new EmptyExpression()).toEsprima();
                    } else {
                        return this.toEsprima_();
                    }
                }

                toCString():string {
                    return this.deleted ? "" : this.toCString_();
                }

                setChildParents() {
                    if (this.deleted) {
                        return;
                    }
                    return this.setChildParents_();
                }

                get children():Node[] {
                    if (this.deleted) {
                        return [];
                    }
                    return this.children_();
                }

                hasChildren():boolean {
                    return this.deleted ? false : this.hasChildren_();
                }

                postOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (this.deleted) {
                        return this;
                    }
                    return this.postOrderTraverse_(visit, data);
                }

                preOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (this.deleted) {
                        return this;
                    }
                    return this.preOrderTraverse_(visit, data);
                }

                inOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (this.deleted) {
                        return this;
                    }
                    return this.inOrderTraverse_(visit, data);
                }

                reversePostOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (this.deleted) {
                        return this;
                    }
                    return this.reversePostOrderTraverse_(visit, data);
                }

                reversePreOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (this.deleted) {
                        return this;
                    }
                    return this.reversePreOrderTraverse_(visit, data);
                }

                toEsprima_():esprima.Syntax.Node {
                    return null
                }

                toCString_():string {
                    return "";
                }

                setChildParents_() {
                    var self = this;
                    if (_.any(this.children, isUndefined)) {
                        debugger;
                    }
                    _.each(this.children, (child) => child.parent = self);
                }

                children_():Node[] {
                    return [];
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class EmptyExpression extends Node {
                constructor() {
                    super("EmptyExpression", unknownLocation(), "", "");
                }

                toEsprima() {
                    return null
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class NullNode extends Node {
                constructor() {
                    super("NullNode", unknownLocation(), "", "");
                }

                toEsprima_():esprima.Syntax.Node {
                    return null;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Literal<T> extends Node {
                value:T;

                constructor(loc:any, raw:string, cform:string, value:T) {
                    super("Literal", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                toEsprima_():esprima.Syntax.Literal {
                    return {
                        type: "Literal",
                        value: this.value,
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                toCString_():string {
                    return this.value.toString();
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class SymbolLiteral extends Literal<string> {
                constructor(loc:any, raw:string, cform:string, value:string) {
                    super(loc, raw, cform, value);
                    this.type = "SymbolLiteral";
                    this.setChildParents();
                }

                /*
                 toEsprima_():esprima.Syntax.Literal {
                 return castTo<esprima.Syntax.Literal>({
                 type: "Identifier",
                 name: this.value,
                 loc: this.loc,
                 raw: this.raw, cform: this.cform
                 })
                 }*/
                static fromCena(o:any):SymbolLiteral {
                    return new SymbolLiteral(o.loc, o.raw, o.cform, o.value);
                }

                toCString_():string {
                    return this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class StringLiteral extends Literal<string> {
                constructor(loc:any, raw:string, cform:string, value:string) {
                    super(loc, raw, cform, value);
                    this.type = "StringLiteral";
                    this.setChildParents();
                }

                static fromCena(o:any):StringLiteral {
                    return new StringLiteral(o.loc, o.raw, o.cform, o.value);
                }

                toCString_():string {
                    return this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class BooleanLiteral extends Literal<boolean> {
                constructor(loc:any, raw:string, cform:string, value:boolean) {
                    super(loc, raw, cform, value);
                    this.type = "BooleanLiteral";
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new BooleanLiteral(o.loc, o.raw, o.cform, o.value);
                }

                toCString_():string {
                    return this.value ? "true" : "false";
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class CharLiteral extends Node {
                value:string

                constructor(loc:any, raw:string, cform:string, value:string) {
                    super("CharLiteral", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                static fromCena(o:any):CharLiteral {
                    return new CharLiteral(o.loc, o.raw, o.cform, o.value);
                }

                toCString_():string {
                    return "'" + this.value + "'";
                }

                toEsprima_():esprima.Syntax.NewExpression {
                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                      builder.identifier(
                        "lib",
                        sloc
                        ),
                        builder.identifier(
                          "c",
                          sloc
                          ),
                          false,
                          sloc
                          );
                          return {
                            type: "NewExpression",
                            loc: this.loc,
                            callee: builder.memberExpression(
                              libc,
                              builder.identifier("Char", sloc),
                              false
                              ),
                        arguments: [
                            castTo<esprima.Syntax.Expression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Integer8Literal extends Node {
                value:number

                constructor(loc:any, raw:string, cform:string, value:number) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                static fromCena(o:any):Integer8Literal {
                    return new Integer8Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima_():esprima.Syntax.NewExpression {
                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                      builder.identifier(
                        "lib",
                        sloc
                        ),
                        builder.identifier(
                          "c",
                          sloc
                          ),
                          false,
                          sloc
                          );
                          return {
                            type: "NewExpression",
                            loc: this.loc,
                            callee: builder.memberExpression(
                              libc,
                              builder.identifier("Int8", sloc),
                              false
                              ),
                        arguments: [
                            castTo<esprima.Syntax.Expression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                toCString_():string {
                    return "" + this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Integer32Literal extends Node {
                value:number

                constructor(loc:any, raw:string, cform:string, value:number) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                static fromCena(o:any):Integer32Literal {
                    return new Integer32Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima_():esprima.Syntax.NewExpression {

                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                      builder.identifier(
                        "lib",
                        sloc
                        ),
                        builder.identifier(
                          "c",
                          sloc
                          ),
                          false,
                          sloc
                          );
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: builder.memberExpression(
                          libc,
                          builder.identifier("Int32", sloc),
                          false
                          ),
                        arguments: [
                            castTo<esprima.Syntax.Expression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                toCString_():string {
                    return "" + this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Integer64Literal extends Node {
                value:string

                constructor(loc:any, raw:string, cform:string, value:string) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                static fromCena(o:any):Integer64Literal {
                    return new Integer64Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima_():esprima.Syntax.NewExpression {

                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                  var libc = builder.memberExpression(
                    builder.identifier(
                      "lib",
                      sloc
                      ),
                      builder.identifier(
                        "c",
                        sloc
                        ),
                        false,
                        sloc
                        );
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: builder.memberExpression(
                          libc,
                          builder.identifier("Int64"),
                          false,
                          sloc
                        ),
                        arguments: [
                            castTo<esprima.Syntax.Expression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                toCString_():string {
                    return "" + this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Float32Literal extends Node {
                value:number

                constructor(loc:any, raw:string, cform:string, value:number) {
                    super("FloatLiteral", loc, raw, cform);
                    this.value = value;
                    this.setChildParents();
                }

                static fromCena(o:any):Float32Literal {
                    return new Float32Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima_():esprima.Syntax.NewExpression {
                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                      builder.identifier(
                        "lib",
                        sloc
                        ),
                        builder.identifier(
                          "c",
                          sloc
                          ),
                          false,
                          sloc
                          );
                          return {
                            type: "NewExpression",
                            loc: this.loc,
                            callee: builder.memberExpression(
                              libc,
                              builder.identifier("Float32", sloc),
                              false
                              ),
                        arguments: [
                            castTo<esprima.Syntax.Expression>({
                                type: "Literal",
                                loc: this.loc,
                                value: this.value
                            })
                        ]
                    }
                }

                toCString_():string {
                    return "" + this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Float64Literal extends Literal<number> {
                constructor(loc:any, raw:string, cform:string, value:number) {
                    super(loc, raw, cform, value);
                    this.type = "Float64Literal";
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new Float64Literal(o.loc, o.raw, o.cform, o.value);
                }

                toCString_():string {
                    return "" + this.value;
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }

            export class ReferenceType extends Node {
                value:Node
                isCUDA:boolean

                constructor(loc:any, raw:string, cform:string, value:any) {
                    super("ReferenceType", loc, raw, cform);
                    this.value = fromCena(value);
                    this.isCUDA = false;
                    this.setChildParents();
                }

                static fromCena(o:any):ReferenceType {
                    return new ReferenceType(o.loc, o.raw, o.cform, o.value);
                }

                makeCUDAReference() {
                    this.isCUDA = true;
                }

                toEsprima_():esprima.Syntax.Comment {
                    return null
                    /* {
                     type: "Comment",
                     value: this.value.toCString(),
                     raw: this.raw,
                     loc: this.loc
                     } */
                }

                toCString_():string {
                    return this.value.toCString() + "*";
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.value.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.value.preOrderTraverse(visit, data);
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.value.inOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.value.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.value.reversePreOrderTraverse(visit, data);
                }
            }
            export class TypeExpression extends Node {
                eaddressSpace:Node[]
                equalifiers:Node[]
                ebases:Node[]
                addressSpace:string[]
                qualifiers:string[]
                bases:string[]

                constructor(loc:any, raw:string, cform:string, addressSpace:string[], qualifiers:string[], bases:string[]) {
                    super("TypeExpression", loc, raw, cform);
                    this.eaddressSpace = _.map(addressSpace || [undefined], (b:Node) => fromCena(b));
                    this.equalifiers = _.map(qualifiers || [undefined], (b:Node) => fromCena(b));
                    this.ebases = _.map(bases || [undefined], (b:Node) => fromCena(b));
                    this.addressSpace = _.map(addressSpace || [undefined], (b:Node) => fromCena(b).toCString());
                    this.qualifiers = _.map(qualifiers || [undefined], (b:Node) => fromCena(b).toCString());
                    this.bases = _.map(bases || [undefined], (b:Node) => fromCena(b).toCString());
                    this.setChildParents();
                }

                static fromCena(o:any):TypeExpression {
                    return new TypeExpression(o.loc, o.raw, o.cform, o.addressSpace, o.qualifiers, o.bases);
                }

                toEsprima_():esprima.Syntax.Comment {
                    return null
                    /* {
                     type: "Comment",
                     value: [this.addressSpace, this.qualifiers, this.bases].join(" "),
                     raw: JSON.stringify({
                     addressSpace: this.addressSpace,
                     qualifiers: this.qualifiers,
                     bases: this.bases
                     }),
                     loc: this.loc
                     } */
                }

                toCString_():string {
                    return _.flatten([this.addressSpace, this.qualifiers, this.bases]).join(" ");
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class Identifier extends Node {
                kind:Node
                name:string

                constructor(loc:any, raw:string, cform:string, name:string, kind?:any) {
                    super("Identifier", loc, raw, cform);
                    this.name = name;
                    if (isUndefined(kind)) {
                        this.kind = new EmptyExpression();
                    } else {
                        this.kind = fromCena(kind);
                    }
                    this.setChildParents();
                }

                static fromCena(o:any):Identifier {
                    return new Identifier(o.loc, o.raw, o.cform, o.name, o.kind);
                }

                makeCUDAReference() {
                    if (this.kind.type === "ReferenceType") {
                        castTo<ReferenceType>(this.kind).makeCUDAReference();
                    }
                }

                toEsprima_():esprima.Syntax.Identifier {
                    if (this.kind.type === "ReferenceType") {

                        var loc = this.loc;
                        var sloc = builder.sourceLocation(
                            builder.position(loc.start.line, loc.start.column),
                            builder.position(loc.end.line, loc.end.column)
                        );
                        var libc = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "c",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        var refname = this.kind.type === "ReferenceType" && castTo<ReferenceType>(this.kind).isCUDA === true ? "cudaReference" : "reference";
                        var ref = builder.memberExpression(libc, builder.identifier(refname, sloc), false, sloc);
                        return castTo<esprima.Syntax.Identifier>({
                            type: "CallExpression",
                            callee: castTo<esprima.Syntax.Identifier>(ref),
                            arguments: [castTo<esprima.Syntax.Identifier>({
                                type: "Identifier",
                                name: "functionStack$",
                                loc: this.loc
                            }),
                                castTo<esprima.Syntax.Identifier>({
                                    type: "Identifier",
                                    name: this.name,
                                    loc: this.loc
                                })],
                            kind: this.kind.toEsprima(),
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        });
                    } else if (_.isObject(this.parent) && this.parent.type !== "Program" && this.parent.type !== "MemberExpression" && this.parent.type !== "BinaryExpression" &&
                        this.parent.type !== "CallExpression" && this.parent.type != "FunctionDeclaration" && this.parent.type !== "FunctionExpression") {
                        var self = this;
                        return builder.memberExpression(builder.identifier("functionStack$", self.loc), builder.identifier(self.name, self.loc), true, self.loc);
                    } else {
                        return {
                            type: "Identifier",
                            name: this.name,
                            kind: this.kind.toEsprima(),
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }
                    }
                }

                toCString_():string {
                    if (false && this.kind.type != "EmptyExpression") {
                        return this.kind.toCString() + " " + this.name;
                    } else {
                        return this.name;
                    }
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    return visit(this, data);
                }
            }
            export class CompoundNode {
                elements:Node[]
                parent:Node
                deleted:boolean;

                constructor(elements:any[]) {
                    this.elements = isUndefined(elements) ? [] : elements.map((elem) => fromCena(elem));
                    var self = this;
                    _.each(this.elements, (elem:Node) => elem.parent = self.parent);
                    this.deleted = false;
                }

                static fromCena(o:any):CompoundNode {
                    return new CompoundNode(o);
                }

                toEsprima():esprima.Syntax.Node[] {
                    if (this.deleted || isUndefined(this.elements)) {
                        return [];
                    } else {
                        return this.elements.map((elem) => elem.toEsprima());
                    }
                }

                get children():Node[] {
                    return this.deleted ? [] : this.elements;
                }

                hasChildren():boolean {
                    return this.deleted ? false : _.isEmpty(this.elements);
                }


                postOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (!this.deleted) {
                        var res:Node;
                        _.each(this.elements, (elem) => res = elem.postOrderTraverse(visit, data));
                        return res;
                    } else {
                        return null;
                    }
                }

                preOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (!this.deleted) {
                        var res:Node;
                        _.each(this.elements, (elem) => res = elem.preOrderTraverse(visit, data));
                        return res;
                    } else {
                        return null;
                    }
                }

                inOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (!this.deleted) {
                        var res:Node;
                        _.each(this.elements, (elem) => res = elem.inOrderTraverse(visit, data));
                        return res;
                    } else {
                        return null;
                    }
                }

                reversePostOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (!this.deleted) {
                        var res:Node;
                        _.forEachRight(this.elements, (elem) => res = elem.reversePostOrderTraverse(visit, data));
                        return res;
                    } else {
                        return null;
                    }
                }

                reversePreOrderTraverse(visit:(Node, any) => Node, data:any):Node {
                    if (!this.deleted) {
                        var res:Node;
                        _.forEachRight(this.elements, (elem) => res = elem.reversePreOrderTraverse(visit, data));
                        return res;
                    } else {
                        return null;
                    }
                }
            }
            function endsWith(subjectString:string, searchString:string, position?) {
                if (position === undefined || position > subjectString.length) {
                    position = subjectString.length;
                }
                position -= searchString.length;
                var lastIndex = subjectString.indexOf(searchString, position);
                return lastIndex !== -1 && lastIndex === position;
            }

            export class BlockStatement extends Node {
                body:CompoundNode

                constructor(loc:any, raw:string, cform:string, body:any) {
                    super("BlockStatement", loc, raw, cform);
                    this.body = new CompoundNode(body);
                    this.setChildParents();
                }

                static fromCena(o:any):BlockStatement {
                    return new BlockStatement(o.loc, o.raw, o.cform, o.body);
                }

                toEsprima_():esprima.Syntax.BlockStatement {
                    var stmts = _.map(this.body.elements,
                        function (elem) {
                            if (elem.type === "EmptyExpression") {
                                return null;
                            } else if (lib.ast.utils.isStatement(elem.toEsprima())) {
                                return elem.toEsprima();
                            } else if (lib.ast.utils.isExpression(elem.toEsprima())) {
                                return {
                                    type: "ExpressionStatement",
                                    expression: elem.toEsprima(),
                                    loc: elem.loc,
                                    raw: elem.raw,
                                    cform: elem.cform
                                }
                            } else {
                                lib.utils.assert.fail(true, "The generated node is neither a statement or expression");
                                return null;
                            }
                        }
                    );
                    stmts = _.reject(stmts, _.isNull);
                    return {
                        type: "BlockStatement",
                        body: castTo<esprima.Syntax.Statement[]>(stmts),
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                toCString_():string {
                    var prog:string[] = _.map(this.body.elements, (elem:Node) => elem.toCString());
                    prog = _.map(prog, (elem:string) => endsWith(elem, ";") ? elem.substring(0, elem.length - 1).trim() : elem.trim());
                    prog = _.filter(prog, (elem:string) => elem !== "");
                    return "{\n" + prog.join(";\n") + "\n}";
                }

                children_():Node[] {
                    return this.body.children;
                }

                hasChildren_():boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.preOrderTraverse(visit, data);
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.reversePostOrderTraverse(visit, data);
                }
            }
            export class FunctionExpression extends Node {
                attributes:string[]
                ret:Node
                id:Identifier
                params:CompoundNode
                body:Node

                constructor(loc:any, raw:string, cform:string, attributes:string[], ret:any, id:any, params:any[], body:any) {
                    super("FunctionExpression", loc, raw, cform);
                    this.attributes = attributes;
                    this.ret = isUndefined(ret) ? new EmptyExpression() : fromCena(ret);
                    this.id = Identifier.fromCena({loc: loc, raw: raw, cform: cform, name: id});
                    this.params = CompoundNode.fromCena(params);
                    if (isUndefined(body)) {
                        this.body = new EmptyExpression();
                    } else if (body.type === "BlockStatement") {
                        var blk:BlockStatement = castTo<BlockStatement>(fromCena(body));
                        this.body = blk;
                    } else {
                        var blk = BlockStatement.fromCena({loc: loc, raw: raw, cform: cform, body: body});
                        this.body = blk;
                    }
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new FunctionExpression(o.loc, o.raw, o.cform, o.attributes, o.ret, o.id, o.params, o.body);
                }

                toEsprima_():esprima.Syntax.FunctionDeclaration {
                    var self = this;
                    var body = self.body.toEsprima();
                    if (body.type === "BlockStatement") {
                        var blk:esprima.Syntax.BlockStatement = castTo<esprima.Syntax.BlockStatement>(body);
                        var threadParams:Node[] = [];
                        if (!_.isEmpty(self.attributes)) {
                            threadParams = [
                                new StringLiteral(self.rloc, "threadIdx", "threadIdx", "threadIdx"),
                                new StringLiteral(self.rloc, "blockIdx", "blockIdx", "blockIdx"),
                                new StringLiteral(self.rloc, "gridIdx", "gridIdx", "gridIdx"),
                                new StringLiteral(self.rloc, "blockDim", "blockDim", "blockDim"),
                                new StringLiteral(self.rloc, "gridDim", "gridDim", "gridDim")
                            ];
                        }
                        var params = threadParams.concat(self.params.elements);
                        var idx = params.length;
                        var self = self;
                        _.eachRight(params,
                            function (param) {
                                var sparam;
                                if (param.type === "StringLiteral") {
                                    sparam = param;
                                } else {
                                    var id:Identifier = castTo<Identifier>(param);
                                    sparam = new StringLiteral(id.rloc, id.raw, id.cform, id.name);
                                }
                                idx--;
                                var k = castTo<esprima.Syntax.ExpressionStatement>({
                                    type: "ExpressionStatement",
                                    expression: {
                                        type: "AssignmentExpression",
                                        operator: "=",
                                        left: {
                                            type: "MemberExpression",
                                            computed: true,
                                            object: {
                                                type: "Identifier",
                                                name: "functionStack$",
                                                loc: self.loc,
                                                raw: self.raw,
                                                cform: self.cform
                                            },
                                            property: sparam.toEsprima(),

                                            loc: self.loc,
                                            raw: self.raw,
                                            cform: self.cform
                                        },
                                        right: {
                                            type: "MemberExpression",
                                            computed: true,
                                            object: {
                                                type: "Identifier",
                                                name: "arguments",
                                                loc: self.loc,
                                                raw: self.raw,
                                                cform: self.cform

                                            },
                                            property: {
                                                type: "Literal",
                                                value: idx,
                                                raw: idx.toString(),
                                                loc: self.loc,
                                                cform: self.cform
                                            }
                                        },
                                        loc: self.loc,
                                        raw: self.raw,
                                        cform: self.cform
                                    },
                                    loc: self.loc,
                                    raw: self.raw,
                                    cform: self.cform
                                });
                                blk.body.unshift(k);
                            }
                        );
                        blk.body.unshift({
                            type: "VariableDeclaration",
                            loc: self.loc,
                            raw: self.raw,
                            cform: self.cform,
                            declarations: [
                                {
                                    type: "VariableDeclarator",
                                    id: {
                                        type: "Identifier",
                                        name: "functionStack$",
                                        loc: self.loc,
                                        raw: self.raw,
                                        cform: self.cform
                                    },
                                    init: {
                                        type: "ObjectExpression",
                                        properties: [],
                                        loc: self.loc,
                                        raw: self.raw,
                                        cform: self.cform
                                    },
                                    loc: self.loc,
                                    raw: self.raw,
                                    cform: self.cform
                                },
                                {
                                    type: "VariableDeclarator",
                                    id: {
                                        type: "Identifier",
                                        name: "functionName$",
                                        loc: self.loc,
                                        raw: self.raw,
                                        cform: self.cform
                                    },
                                    init: {
                                        type: "Literal",
                                        value: self.id.name,
                                        loc: self.loc,
                                        raw: self.raw,
                                        cform: self.cform
                                    },
                                    loc: self.loc,
                                    raw: self.raw,
                                    cform: self.cform
                                }
                            ],
                            kind: "var"
                        });
                    }
                    if (!_.isEmpty(self.attributes)) {
                        return castTo<esprima.Syntax.FunctionDeclaration >({
                            type: "FunctionDeclaration",
                            id: castTo<esprima.Syntax.Identifier>(self.id.toEsprima()),
                            params: [],
                            body: builder.blockStatement([
                                    {
                                        type: "FunctionDeclaration",
                                        id: builder.identifier(self.id.name + "_", self.id.loc),
                                        params: [],
                                        body: castTo<esprima.Syntax.BlockStatementOrExpression>(body),
                                        ret: self.ret.toEsprima(),
                                        attributes: self.attributes,
                                        defaults: [],
                                        generator: false,
                                        expression: false,
                                        raw: self.raw, cform: self.cform,
                                        loc: self.loc
                                    },
                                    builder.variableDeclaration("var", [
                                            builder.variableDeclarator(
                                                builder.identifier("threadIdx", self.loc),
                                                builder.objectExpression(
                                                    _.map(["x", "y", "z"], (dim) =>
                                                        builder.property("init", builder.identifier(dim, self.loc), builder.literal(0, self.loc), self.loc))
                                                ),
                                                self.loc
                                            ),
                                            builder.variableDeclarator(
                                                builder.identifier("blockIdx", self.loc),
                                                builder.memberExpression(builder.identifier("argument", self.loc), builder.literal(0, self.loc), true, self.loc),
                                                self.loc
                                            ),
                                            builder.variableDeclarator(
                                                builder.identifier("blockDim", self.loc),
                                                builder.memberExpression(builder.identifier("argument", self.loc), builder.literal(1, self.loc), true, self.loc),
                                                self.loc
                                            ),
                                            builder.variableDeclarator(
                                                builder.identifier("gridIdx", self.loc),
                                                builder.memberExpression(builder.identifier("argument", self.loc), builder.literal(2, self.loc), true, self.loc),
                                                self.loc
                                            ),
                                            builder.variableDeclarator(
                                                builder.identifier("gridDim", self.loc),
                                                builder.memberExpression(builder.identifier("argument", self.loc), builder.literal(3, self.loc), true, self.loc),
                                                self.loc
                                            )
                                        ],
                                        self.loc
                                    ),
                                    builder.variableDeclaration("var",
                                        _.map(this.params.elements, (param, idx) =>
                                                builder.variableDeclarator(
                                                    param.toEsprima(),
                                                    builder.memberExpression(builder.identifier("argument", self.loc), builder.literal(3 + idx, self.loc), true, self.loc),
                                                    self.loc
                                                )
                                        ),
                                        self.loc
                                    ),
                                    builder.forStatement(
                                        builder.assignmentExpression(
                                            "=",
                                            builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                builder.literal("z", self.loc), true, self.loc),
                                            builder.literal(0, self.loc),
                                            self.loc
                                        ),
                                        builder.binaryExpression("<",
                                            builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                builder.literal("z", self.loc), true, self.loc),
                                            builder.logicalExpression("||",
                                                builder.memberExpression(builder.identifier("blockDim", self.loc),
                                                    builder.literal("z", self.loc), true, self.loc),
                                                builder.literal(1, self.loc)),
                                            self.loc
                                        ),
                                        builder.updateExpression("++", builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                            builder.literal("z", self.loc), true, self.loc), false, self.loc),
                                        builder.forStatement(
                                            builder.assignmentExpression(
                                                "=",
                                                builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                    builder.literal("y", self.loc), true, self.loc),
                                                builder.literal(0, self.loc),
                                                self.loc
                                            ),
                                            builder.binaryExpression("<",
                                                builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                    builder.literal("y", self.loc), true, self.loc),
                                                builder.logicalExpression("||",
                                                    builder.memberExpression(builder.identifier("blockDim", self.loc),
                                                        builder.literal("y", self.loc), true, self.loc),
                                                    builder.literal(1, self.loc)),
                                                self.loc
                                            ),
                                            builder.updateExpression("++", builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                builder.literal("y", self.loc), true, self.loc), false, self.loc),

                                            builder.forStatement(
                                                builder.assignmentExpression(
                                                    "=",
                                                    builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                        builder.literal("x", self.loc), true, self.loc),
                                                    builder.literal(0, self.loc),
                                                    self.loc
                                                ),
                                                builder.binaryExpression("<",
                                                    builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                        builder.literal("x", self.loc), true, self.loc),
                                                    builder.logicalExpression("||",
                                                        builder.memberExpression(builder.identifier("blockDim", self.loc),
                                                            builder.literal("x", self.loc), true, self.loc),
                                                        builder.literal(1, self.loc)),
                                                    self.loc
                                                ),
                                                builder.updateExpression("++", builder.memberExpression(builder.identifier("threadIdx", self.loc),
                                                    builder.literal("x", self.loc), true, self.loc), false, self.loc),
                                                builder.blockStatement([
                                                    builder.expressionStatement(
                                                        builder.callExpression(
                                                            builder.identifier(self.id.name + "_", self.id.loc),
                                                            _.map(["threadIdx", "blockIdx", "gridIdx", "blockDim", "gridDim"], (fld) => builder.identifier(fld, self.loc)).concat(this.params.toEsprima()),
                                                            self.loc
                                                        ), self.loc)], self.loc)
                                            )
                                        )
                                    )
                                    /*

                                     builder.forStatement(
                                     builder.variableDeclaration("var",
                                     builder.variableDeclarator(),
                                     self.loc
                                     ),
                                     builder.binaryExpression()
                                     )
                                     */
                                ],
                                self.loc
                            ),
                            ret: self.ret.toEsprima(),
                            attributes: self.attributes,
                            defaults: [],
                            generator: false,
                            expression: false,
                            raw: self.raw, cform: self.cform,
                            loc: self.loc
                        })
                    } else {
                        return castTo<esprima.Syntax.FunctionDeclaration >({
                            type: "FunctionDeclaration",
                            id: castTo<esprima.Syntax.Identifier>(this.id.toEsprima()),
                            params: [],
                            body: castTo<esprima.Syntax.BlockStatementOrExpression>(body),
                            ret: this.ret.toEsprima(),
                            attributes: this.attributes,
                            defaults: [],
                            generator: false,
                            expression: false,
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        })
                    }
                }

                toCString_():string {
                    return [this.attributes].join(" ") + this.ret.toCString() + " " + this.id.toCString() + " (" + _.map(this.params.elements, (p:Node) => p.toCString()).join(", ") + ") " + this.body.toCString();
                }

                children_():Node[] {
                    return _.flatten<Node>([this.body, this.ret, this.id, this.params.children]);
                }

                hasChildren_():boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.ret.postOrderTraverse(visit, data);
                    this.id.postOrderTraverse(visit, data);
                    this.params.postOrderTraverse(visit, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.ret.preOrderTraverse(visit, data);
                    this.id.preOrderTraverse(visit, data);
                    this.params.preOrderTraverse(visit, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.ret.inOrderTraverse(visit, data);
                    this.id.inOrderTraverse(visit, data);
                    this.params.inOrderTraverse(visit, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    this.params.reversePostOrderTraverse(visit, data);
                    this.id.reversePostOrderTraverse(visit, data);
                    this.ret.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    this.params.reversePreOrderTraverse(visit, data);
                    this.id.reversePreOrderTraverse(visit, data);
                    this.ret.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class CallExpression extends Node {
                callee:Identifier
                arguments:CompoundNode
                config:Node[]
                isCUDA:boolean = false

                constructor(loc:any, raw:string, cform:string, callee:any, arguments:any[], config?:any) {
                    super("CallExpression", loc, raw, cform);
                    if (lib.utils.isString(callee)) {
                        this.callee = new Identifier(this.loc, callee, callee, callee);
                    } else {
                        this.callee = Identifier.fromCena(callee);
                    }
                    this.arguments = new CompoundNode(arguments);
                    this.config = isUndefined(config) ? [] : _.map(config, (c:any) => fromCena(c));
                    this.isCUDA = !isUndefined(config);
                    if (this.callee.name === "sizeof") {
                        var c:TypeExpression = castTo<TypeExpression>(fromCena(arguments[0]));
                        this.arguments = new CompoundNode([c.ebases[0]])
                    }
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new CallExpression(o.loc, o.raw, o.cform, o.callee, castTo<any[]>(o.arguments), o.config);
                }


                toEsprima_():esprima.Syntax.CallExpression {
                    var self = this;
                    var callee = this.callee.toEsprima();
                    var args:any = this.arguments.elements;
                    var loc = this.callee.loc;
                    var sloc = builder.sourceLocation(
                        builder.position(loc.start.line, loc.start.column),
                        builder.position(loc.end.line, loc.end.column)
                    );
                    if (this.config.length > 0) {
                        _.each(args, (arg : any) => {
                            if (_.isFunction(arg.makeCUDAReference)) { arg.makeCUDAReference(); }
                        });
                    }
                    if (startsWith(this.callee.name, "wb")) {
                        var libwb = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "wb",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        callee = builder.memberExpression(
                            libwb,
                            callee,
                            false,
                            sloc
                        );
                    } else if (startsWith(this.callee.name, "cuda")) {
                        var libcuda = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "cuda",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        callee = builder.memberExpression(
                            libcuda,
                            callee,
                            false,
                            sloc
                        );
                        if (_.any(args, (arg:Node) => arg.type === "ReferenceExpression" || (arg.type === "Identifier" && castTo<Identifier>(arg).kind.type === "ReferenceType"))) {
                            if (this.callee.name === "cudaMalloc") {
                                castTo<ReferenceExpression>(args[0]).makeCUDAReference();
                            } else if (this.callee.name === "cudaMemcpy") {
                                if (castTo<Identifier>(_.last(args)).name === "cudaMemcpyHostToDevice") {
                                    castTo<ReferenceExpression>(args[0]).makeCUDAReference();
                                } else {
                                    castTo<ReferenceExpression>(args[1]).makeCUDAReference();
                                }
                            } else if (this.callee.name === "cudaFree") {
                                castTo<ReferenceExpression>(args[0]).makeCUDAReference();
                            }
                        }
                    } else if (_.contains(["malloc", "free", "sizeof"], this.callee.name)) {
                        var libc = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "c",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        callee = builder.memberExpression(
                            libc,
                            callee,
                            false,
                            sloc
                        );
                    } else if (_.contains(["ceil", "floor"], this.callee.name)) {
                        var libm = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "m",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        callee = builder.memberExpression(
                            libm,
                            callee,
                            false,
                            sloc
                        );
                    }
                    if (this.config.length > 0) {
                        return builder.blockStatement(_.flatten([
                            builder.variableDeclaration("var",
                                _.map(["gridDim$", "blockDim$"], (d, cn) => builder.variableDeclarator(
                                    builder.identifier(d, self.loc),
                                    builder.objectExpression(
                                        _.map(["x", "y", "z"], (dim, idx) => {
                                                var id = builder.memberExpression(builder.identifier("functionStack$", self.loc), builder.literal(castTo<Identifier>(self.config[cn]).name, self.loc), true, self.loc);
                                                return builder.property("init", builder.identifier(dim, self.loc), builder.conditionalExpression(
                                                    builder.binaryExpression(
                                                        "<",
                                                        builder.memberExpression(id, builder.identifier("length", self.loc), false, self.loc),
                                                        builder.literal(idx, self.loc),
                                                        self.loc
                                                    ),
                                                    builder.memberExpression(id, builder.literal(idx, self.loc), true, self.loc),
                                                    builder.literal(1, self.loc),
                                                    self.loc
                                                ));
                                            }
                                        ),
                                        self.loc
                                    ),
                                    self.loc
                                )),
                                self.loc
                            ),
                            _.reduceRight(["gridIdxZ", "gridIdxY", "gridIdxX", "blockIdxZ", "blockIdxY", "blockIdxX"],
                            function(res, id) {
                              return builder.forStatement(
                                builder.variableDeclaration(
                                  "var",
                                  [
                                  builder.variableDeclarator(
                                    builder.identifier(id, self.loc),
                                    builder.literal(0, self.loc)
                                    )],
                                  self.loc
                                  ),
                                builder.binaryExpression(
                                  "<",
                                  builder.identifier(id, self.loc),
                                  builder.memberExpression(
                                    builder.identifier(startsWith(id, "grid") ? "gridDim$" : "blockDim$", self.loc),
                                    builder.identifier(id[id.length-1].toLowerCase(), self.loc),
                                    false,
                                    self.loc
                                  ),
                                  self.loc
                                  ),
                                  builder.updateExpression("++", builder.identifier(id, self.loc), true, self.loc),
                                  builder.blockStatement(res ? [res] : [], self.loc),
                                self.loc
                                )
                              },
                              builder.expressionStatement(
                                builder.callExpression(
                                  builder.memberExpression(
                                    builder.memberExpression(
                                      builder.identifier(
                                        "lib",
                                        self.loc
                                        ),
                                        builder.identifier(
                                          "parallel",
                                          self.loc
                                          ),
                                          false,
                                          self.loc
                                          ),
                                          builder.identifier(
                                            "scheduleThread",
                                            self.loc
                                            ),
                                            false,
                                            self.loc
                                            ),
                                            [
                                            builder.callExpression(
                                              builder.functionExpression(null, [builder.identifier("gridIdx$", self.loc), builder.identifier("blockIdx$", self.loc)],
                                              builder.functionExpression(null, [],
                                                builder.blockStatement([
                                                  builder.expressionStatement(
                                                    builder.callExpression(
                                                      this.callee.toEsprima(),
                                                      _.map(["blockIdx$", "blockDim$", "gridIdx$", "gridDim$"], (b) => builder.identifier(b, self.loc)).concat(_.map(args, (a : Node) => a.toEsprima()))
                                                      ),
                                                      self.loc
                                                      )]),
                                                      self.loc
                                                      ),
                                                      self.loc
                                                      ),
                                                      ["gridDim", "blockDim"].map((gb) =>
                                                      builder.objectExpression(["x", "y", "z"].map((dim : string) =>
                                                      builder.property("init", builder.identifier(dim, self.loc), builder.identifier(gb + dim.toUpperCase(), self.loc), self.loc)
                                                      ), self.loc)
                                                      )
                                                      )
                                                      ],
                                                      self.loc
                                                      ),
                                                      self.loc
                                                      )
                        )]), self.loc);
                    } else {
                        args = _.map(this.config.concat(args), (a:Node) => a.toEsprima());
                        return castTo<esprima.Syntax.CallExpression >({
                            type: "CallExpression",
                            config: _.map(this.config, (c:Node) => c.toEsprima()),
                            isCUDA: this.isCUDA,
                            callee: castTo<esprima.Syntax.Expression>(callee),
                            arguments: args,
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        })
                    }
                }

                toCString_():string {
                    var ret:string = this.callee.toCString();
                    if (this.isCUDA) {
                        ret += "<<<" + _.map(this.config, (c:Node) => c.toCString()).join(", ") + ">>>";
                    }
                    ret += " (" + _.map(this.arguments.elements, (p:Node) => p.toCString()).join(", ") + ") ";
                    return ret;
                }

                children_():Node[] {
                    return _.flatten<Node>([this.callee, this.arguments.children]);
                }

                hasChildren_():boolean {
                    return false;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.callee.postOrderTraverse(visit, data);
                    _.each(this.config, (c:Node) => c.postOrderTraverse(visit, data));
                    return this.arguments.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.callee.preOrderTraverse(visit, data);
                    _.each(this.config, (c:Node) => c.postOrderTraverse(visit, data));
                    this.arguments.preOrderTraverse(visit, data);
                    return visit(this, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.callee.inOrderTraverse(visit, data);
                    _.each(this.config, (c:Node) => c.inOrderTraverse(visit, data));
                    return this.arguments.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.arguments.reversePostOrderTraverse(visit, data);
                    _.each(this.config, (c:Node) => c.reversePostOrderTraverse(visit, data));
                    this.callee.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.arguments.reversePreOrderTraverse(visit, data);
                    _.each(this.config, (c:Node) => c.reversePreOrderTraverse(visit, data));
                    this.callee.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ParenExpression extends Node {
                expression:Node

                constructor(loc:any, raw:string, cform:string, expression:any) {
                    super("ParenExpression", loc, raw, cform);
                    this.expression = fromCena(expression);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ParenExpression(o.loc, o.raw, o.cform, o.expression);
                }

                toEsprima_():esprima.Syntax.Node {
                    return castTo<esprima.Syntax.Expression>(this.expression.toEsprima());
                }

                toCString_():string {
                    return "(" + this.expression.toCString() + ")";
                }

                children_():Node[] {
                    return [this.expression];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.expression.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.expression.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class DereferenceExpression extends Node {
                argument:Node
                rawArgument:any

                constructor(loc:any, raw:string, cform:string, argument:any) {
                    super("DereferenceExpression", loc, raw, cform);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new DereferenceExpression(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima_():esprima.Syntax.CallExpression {
                    var call:CallExpression = new CallExpression(this.loc, this.raw, this.cform, new Identifier(this.loc, this.raw, this.cform, "dereference"), [this.rawArgument]);
                    return castTo<esprima.Syntax.CallExpression>(call.toEsprima());
                }

                toCString_():string {
                    return "&" + this.argument.toCString();
                }

                children_():Node[] {
                    return [this.argument];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ReferenceExpression extends Node {
                argument:Node
                rawArgument:any
                isCUDA:boolean

                constructor(loc:any, raw:string, cform:string, argument:any) {
                    super("ReferenceExpression", loc, raw, cform);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                    this.isCUDA = false;
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ReferenceExpression(o.loc, o.raw, o.cform, o.argument);
                }

                makeCUDAReference() {
                    if (this.argument.type === "ReferenceExpression") {
                        castTo<ReferenceExpression>(this.argument).makeCUDAReference();
                    } else if (this.argument.type === "Identifier" && castTo<Identifier>(this.argument).kind.type === "ReferenceType") {
                        castTo<Identifier>(this.argument).makeCUDAReference();
                    } else {
                        this.isCUDA = true;
                    }
                }

                toEsprima_():esprima.Syntax.CallExpression {
                    var loc = this.loc;
                    var sloc = builder.sourceLocation(
                        builder.position(loc.start.line, loc.start.column),
                        builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                        builder.identifier(
                            "lib",
                            sloc
                        ),
                        builder.identifier(
                            "c",
                            sloc
                        ),
                        false,
                        sloc
                    );
                    var refname = this.isCUDA ? "cudaReference" : "reference";
                    var ref = builder.memberExpression(libc, builder.identifier(refname, sloc), false, sloc);
                    return builder.callExpression(ref, [builder.identifier("functionStack$", sloc), this.argument.toEsprima()], sloc);
                }

                toCString_():string {
                    return "*" + this.argument.toCString();
                }

                children_():Node[] {
                    return [this.argument];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class UnaryExpression extends Node {
                operator:string
                rawArgument:any
                argument:Node

                constructor(loc:any, raw:string, cform:string, operator:string, argument:any) {
                    super("UnaryExpression", loc, raw, cform);
                    this.operator = operator
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    if (isUndefined(o.operator)) {
                        return new ErrorNode("Invalid UnaryExpression");
                    } else if (o.operator === "*") {
                        return DereferenceExpression.fromCena(o);
                    } else if (o.operator === "&") {
                        return ReferenceExpression.fromCena(o);
                    } else {
                        return new UnaryExpression(o.loc, o.raw, o.cform, o.operator, o.argument);
                    }
                }

                toEsprima_():any {
                    if (this.operator === "*") {
                        var dnd = new DereferenceExpression(this.loc, this.raw, this.cform, this.rawArgument);
                        return dnd.toEsprima();
                    } else if (this.operator === "&") {
                        var rnd = new ReferenceExpression(this.loc, this.raw, this.cform, this.rawArgument);
                        return rnd.toEsprima();
                    } else if (this.operator === "++" || this.operator === "--") {
                        return builder.updateExpression(this.operator, this.argument.toEsprima(), true, this.loc);
                    }
                    return {
                        type: "UnaryExpression",
                        argument: this.argument.toEsprima(),
                        operator: this.operator,
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                toCString_():string {
                    return this.operator + this.argument.toCString();
                }

                children_():Node[] {
                    return [this.argument];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            function makeOp(op:string, symbol:string):Identifier {
                return new Identifier(unknownLocation(), symbol, symbol, op);
            }

            export class BinaryExpression extends Node {
                operator:string
                right:Node
                left:Node
                static PropertyTable:{
                    [key: string]: Identifier;
                } = undefined;

                constructor(loc:any, raw:string, cform:string, operator:string, right:any, left:any) {
                    super("BinaryExpression", loc, raw, cform);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                    if (BinaryExpression.PropertyTable === undefined) {
                        BinaryExpression.PropertyTable = {
                            "+": makeOp("plus", "+"),
                            "-": makeOp("minus", "-"),
                            "*": makeOp("times", "*"),
                            "/": makeOp("divide", "/"),
                            "%": makeOp("mod", "%"),
                            "&&": makeOp("and", "&&"),
                            "||": makeOp("or", "||"),
                            "==": makeOp("equals", "=="),
                            "!=": makeOp("notequals", "!="),
                            ">": makeOp("greaterthan", ">"),
                            ">=": makeOp("greaterthanequals", ">="),
                            "<": makeOp("lessthan", "<"),
                            "<=": makeOp("lessthanequals", "<="),
                            "|": makeOp("bitor", "|"),
                            "&": makeOp("bitand", "&"),
                            "^": makeOp("bitxor", "^"),
                            ">>": makeOp("shiftright", ">>"),
                            "<<": makeOp("shiftleft", "<<")
                        };
                    }
                    if (this.left.toEsprima().type === "ExpressionStatement") {
                        debugger;
                    }
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new BinaryExpression(o.loc, o.raw, o.cform, o.operator, o.left, o.right);
                }

                get property():Identifier {
                    return BinaryExpression.PropertyTable[this.operator];
                }

                toEsprima_():esprima.Syntax.CallExpression {
                    var method:esprima.Syntax.MemberExpression = {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.Expression>(this.left.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property),
                        computed: false, /* A member expression. If computed === true,
                         the node corresponds to a computed e1[e2] expression and property is an Expression.
                         If computed === false, the node corresponds to a static e1.x expression and
                         property is an Identifier.*/
                        loc: this.left.loc,
                        raw: this.left.raw, cform: this.left.cform
                    };
                    return {
                        type: "CallExpression",
                        callee: castTo<esprima.Syntax.Expression>(method),
                        arguments: castTo<esprima.Syntax.Expression[]>([this.right.toEsprima()]),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                toCString_():string {
                    return this.left.toCString() + " " + this.operator + " " + this.right.toCString();
                }

                children_():Node[] {
                    return [this.left, this.right];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.postOrderTraverse(visit, data);
                    this.right.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.left.preOrderTraverse(visit, data);
                    return this.right.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.inOrderTraverse(visit, data);
                    visit(this, data);
                    return this.right.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.left.reversePostOrderTraverse(visit, data);
                    return this.right.reversePostOrderTraverse(visit, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.right.reversePreOrderTraverse(visit, data);
                    this.left.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }

            export class UndefinedExpression extends Node {

                constructor() {
                    super("VariableDeclarator", unknownLocation(), "", "");
                }

                toEsprima_():esprima.Syntax.Node {
                    return ({type: "Identifier", name: "undefined"});
                }
            }

            export class VariableDeclarator extends Node {
                init:Node
                id:Identifier
                kind:Node

                constructor(loc:any, raw:string, cform:string, init:any, id:any) {
                    super("VariableDeclarator", loc, raw, cform);
                    this.init = isUndefined(init) ? new UndefinedExpression() : fromCena(init);
                    this.id = castTo<Identifier>(fromCena(id));
                    this.kind = this.id.kind;
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new VariableDeclarator(o.loc, o.raw, o.cform, o.init, o.id);
                }

                toEsprima_():esprima.Syntax.Node {
                    if (this.kind.type === "ReferenceType") {
                        var loc = this.loc;
                        var sloc = builder.sourceLocation(
                            builder.position(loc.start.line, loc.start.column),
                            builder.position(loc.end.line, loc.end.column)
                        );
                        var libc = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "c",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        return builder.expressionStatement(
                            builder.assignmentExpression("=", builder.identifier(this.id.name, sloc),
                                builder.callExpression(
                                    builder.memberExpression(libc, builder.identifier("makeReference", sloc), false, sloc),
                                    [builder.literal(this.id.name, sloc)].concat(this.init.toEsprima()),
                                    sloc
                                ), sloc),
                            sloc
                        );
                    } else {
                        var id = {
                            type: "MemberExpression",
                            computed: true,
                            object: castTo<esprima.Syntax.Identifier>({
                                type: "Identifier",
                                name: "functionStack$",
                                raw: this.raw, cform: this.cform,
                                loc: this.loc
                            }),
                            property: builder.literal(this.id.name, this.id.loc),
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        };
                        return {
                            type: "ExpressionStatement",
                            expression: {
                                type: "AssignmentExpression",
                                operator: "=",
                                right: castTo<esprima.Syntax.Expression>(this.init.toEsprima()),
                                left: id,
                                raw: this.raw, cform: this.cform,
                                loc: this.loc
                            },
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }
                    }
                    /* else {
                     return {
                     type: "VariableDeclarator",
                     init: castTo<esprima.Syntax.Expression>(this.init.toEsprima()),
                     id: castTo<esprima.Syntax.Identifier>(this.id.toEsprima()),
                     raw: this.raw, cform: this.cform,
                     loc: this.loc
                     }
                     } */
                }

                toCString_():string {
                    if (this.init.type != "EmptyExpression") {
                        return this.kind.toCString() + " " + this.id.toCString() + " = " + this.init.toCString();
                    } else {
                        return this.kind.toCString() + " " + this.id.toCString();
                    }
                }

                children_():Node[] {
                    return [this.init, this.id];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.id.postOrderTraverse(visit, data);
                    this.init.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.id.preOrderTraverse(visit, data);
                    return this.init.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.id.inOrderTraverse(visit, data);
                    this.init.inOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.id.reversePostOrderTraverse(visit, data);
                    return this.init.reversePostOrderTraverse(visit, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.init.reversePreOrderTraverse(visit, data);
                    return this.id.reversePreOrderTraverse(visit, data);
                }
            }

            export class VariableDeclaration extends Node {
                declarations:Node[]

                constructor(loc:any, raw:string, cform:string, declarations:any) {
                    super("VariableDeclaration", loc, raw, cform);
                    this.declarations = _.map(declarations, fromCena);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new VariableDeclaration(o.loc, o.raw, o.cform, o.declarations);
                }

                toEsprima_():esprima.Syntax.Node {
                    if (this.declarations.length === 1) {
                        return this.declarations[0].toEsprima();
                    } else {
                        return {
                            type: "SequenceExpression",
                            expressions: castTo<esprima.Syntax.Node[]>(this.declarations.map((decl) => decl.toEsprima())),
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }
                    }
                }

                toCString_():string {
                    return _.map(this.declarations, (decl:Node) => decl.toCString()).join(", ");
                }

                children_():Node[] {
                    return this.declarations;
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    _.each(this.declarations, (decl) => decl.postOrderTraverse(visit, data));
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    var res;
                    visit(this, data);
                    _.each(this.declarations, (decl) => res = decl.preOrderTraverse(visit, data));
                    return res;
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    _.each(this.declarations, (decl) => decl.inOrderTraverse(visit, data));
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    var res;
                    visit(this, data);
                    _.eachRight(this.declarations, (decl) => res = decl.reversePostOrderTraverse(visit, data));
                    return res;
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    var res;
                    _.eachRight(this.declarations, (decl) => res = decl.reversePreOrderTraverse(visit, data));
                    return visit(this, data);
                }
            }
            export class AssignmentExpression extends Node {
                operator:string
                right:Node
                left:Node

                constructor(loc:any, raw:string, cform:string, operator:string, left:any, right:any) {
                    super("AssignmentExpression", loc, raw, cform);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new AssignmentExpression(o.loc, o.raw, o.cform, o.operator, o.left, o.right);
                }

                toEsprima_():esprima.Syntax.AssignmentExpression {
                    if (this.left.type === "Identifier" && castTo<Identifier>(this.left).kind.type === "ReferenceType") {
                        var left:Identifier = castTo<Identifier>(this.left);

                        var loc = this.loc;
                        var sloc = builder.sourceLocation(
                            builder.position(loc.start.line, loc.start.column),
                            builder.position(loc.end.line, loc.end.column)
                        );
                        var libc = builder.memberExpression(
                            builder.identifier(
                                "lib",
                                sloc
                            ),
                            builder.identifier(
                                "c",
                                sloc
                            ),
                            false,
                            sloc
                        );
                        var lefte = builder.memberExpression(
                            builder.identifier("functionStack$", left.loc),
                            builder.literal(left.name, left.loc),
                            true,
                            left.loc
                        );
                        return builder.expressionStatement(
                            builder.assignmentExpression("=", lefte,
                                builder.callExpression(
                                    builder.memberExpression(libc, builder.identifier("makeReference", sloc), false, sloc),
                                    [builder.literal(left.name, sloc)].concat(this.right.toEsprima()),
                                    sloc
                                ), sloc),
                            sloc
                        );
                    } else {
                        return {
                            type: "AssignmentExpression",
                            operator: this.operator,
                            left: castTo<esprima.Syntax.Expression>(this.left.toEsprima()),
                            right: castTo<esprima.Syntax.Expression>(this.right.toEsprima()),
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }
                    }
                }

                toCString_():string {
                    return this.left.toCString() + " " + this.operator + " " + this.right.toCString();
                }

                children_():Node[] {
                    return [this.left, this.right];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.postOrderTraverse(visit, data);
                    this.right.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.left.preOrderTraverse(visit, data);
                    return this.right.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.inOrderTraverse(visit, data);
                    visit(this, data);
                    return this.right.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.left.reversePostOrderTraverse(visit, data);
                    return this.right.reversePostOrderTraverse(visit, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.right.reversePreOrderTraverse(visit, data);
                    this.left.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class IfStatement extends Node {
                test:Node
                consequent:Node
                alternate:Node

                constructor(loc:any, raw:string, cform:string, test:any, consequent:any, alternate?:any) {
                    super("IfStatement", loc, raw, cform);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = isUndefined(alternate) ? new EmptyExpression() : fromCena(alternate);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new IfStatement(o.loc, o.raw, o.cform, o.test, o.consequent, o.alternate);
                }

                toEsprima_():esprima.Syntax.IfStatement {
                    return {
                        type: "IfStatement",
                        test: castTo<esprima.Syntax.Expression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.Statement>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.Statement>(this.consequent.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                toCString_():string {
                    var ret:string = "if (" + this.test.toCString() + ") " + this.consequent.toCString() + " ";
                    if (this.alternate.type != "EmptyExpression") {
                        ret += " else " + this.alternate.toCString();
                    }
                    return ret;
                }

                children_():Node[] {
                    return [this.test, this.consequent, this.alternate];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.test.postOrderTraverse(visit, data);
                    this.alternate.postOrderTraverse(visit, data);
                    this.consequent.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.test.preOrderTraverse(visit, data);
                    this.alternate.preOrderTraverse(visit, data);
                    return this.consequent.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.test.inOrderTraverse(visit, data);
                    this.alternate.inOrderTraverse(visit, data);
                    return this.consequent.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.consequent.reversePostOrderTraverse(visit, data);
                    this.alternate.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.consequent.reversePreOrderTraverse(visit, data);
                    this.alternate.reversePreOrderTraverse(visit, data);
                    return this.test.reversePreOrderTraverse(visit, data);
                }
            }
            export class ConditionalExpression extends Node {
                test:Node
                consequent:Node
                alternate:Node

                constructor(loc:any, raw:string, cform:string, test:any, consequent:any, alternate?:any) {
                    super("ConditionalExpression", loc, raw, cform);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = fromCena(alternate);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ConditionalExpression(o.loc, o.raw, o.cform, o.test, o.consequent, o.alternate);
                }

                toEsprima_():esprima.Syntax.ConditionalExpression {
                    // debugger;
                    return {
                        type: "ConditionalExpression",
                        test: castTo<esprima.Syntax.Expression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.Expression>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.Expression>(this.consequent.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                toCString_():string {
                    return this.test.toCString() + " ? " + this.consequent.toCString() + " : " + this.alternate.toCString();
                }

                children_():Node[] {
                    return [this.test, this.consequent, this.alternate];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.test.postOrderTraverse(visit, data);
                    this.alternate.postOrderTraverse(visit, data);
                    this.consequent.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.test.preOrderTraverse(visit, data);
                    this.alternate.preOrderTraverse(visit, data);
                    return this.consequent.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.test.inOrderTraverse(visit, data);
                    this.alternate.inOrderTraverse(visit, data);
                    return this.consequent.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.consequent.reversePostOrderTraverse(visit, data);
                    this.alternate.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.consequent.reversePreOrderTraverse(visit, data);
                    this.alternate.reversePreOrderTraverse(visit, data);
                    return this.test.reversePreOrderTraverse(visit, data);
                }
            }
            export class ForStatement extends Node {
                init:Node
                test:Node
                update:Node
                body:Node

                constructor(loc:any, raw:string, cform:string, init:any, test:any, update:any, body:any) {
                    super("ForStatement", loc, raw, cform);
                    this.init = fromCena(init);
                    this.test = fromCena(test);
                    this.update = fromCena(update);
                    this.body = fromCena(body);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ForStatement(o.loc, o.raw, o.cform, o.init, o.test, o.update, o.body);
                }

                toEsprima_():esprima.Syntax.ForStatement {
                    return {
                        type: "ForStatement",
                        init: castTo<esprima.Syntax.VariableDeclaratorOrExpression>(this.init.toEsprima()),
                        test: castTo<esprima.Syntax.Expression>(this.test.toEsprima()),
                        update: castTo<esprima.Syntax.Expression>(this.update.toEsprima()),
                        body: castTo<esprima.Syntax.Statement>(this.body.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                children_():Node[] {
                    return [this.init, this.test, this.update, this.body];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.init.postOrderTraverse(visit, data);
                    this.test.postOrderTraverse(visit, data);
                    this.update.postOrderTraverse(visit, data);
                    this.body.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.init.preOrderTraverse(visit, data);
                    this.test.preOrderTraverse(visit, data);
                    this.update.preOrderTraverse(visit, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.init.inOrderTraverse(visit, data);
                    this.test.inOrderTraverse(visit, data);
                    this.update.inOrderTraverse(visit, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    this.update.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    this.init.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.body.reversePostOrderTraverse(visit, data);
                    this.update.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return this.init.reversePostOrderTraverse(visit, data);
                }

            }
            export class ProgramExpression extends Node {
                body:CompoundNode;

                constructor(loc:any, raw:string, cform:string, body:any[]) {
                    super("ProgramExpression", loc, raw, cform);
                    this.body = new CompoundNode(body);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ProgramExpression(o.loc, o.raw, o.cform, o.body);
                }

                toEsprima_():esprima.Syntax.Program {
                    return {
                        type: "Program",
                        body: castTo<esprima.Syntax.Statement[]>(this.body.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }


                toCString_():string {
                    return _.map(this.body.elements, (elem:Node) => elem.toCString()).join("\n");
                }

                children_():Node[] {
                    return this.body.children;
                }

                hasChildren_():boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ReturnStatement extends Node {
                argument:Node;

                constructor(loc:any, raw:string, cform:string, argument?:Node) {
                    super("ReturnStatement", loc, raw, cform);
                    if (argument) {
                        this.argument = fromCena(argument);
                    } else {
                        this.argument = new EmptyExpression();
                    }
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ReturnStatement(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima_():esprima.Syntax.ReturnStatement {

                    return {
                        type: "ReturnStatement",
                        loc: this.loc,
                        raw: this.raw, cform: this.cform,
                        argument: lib.utils.castTo<esprima.Syntax.Expression>(
                            this.argument.toEsprima()
                        )
                    }
                }

                children_():Node[] {
                    return [this.argument];
                }

                hasChildren_():boolean {
                    return !(this.argument instanceof EmptyExpression);
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ExpressionStatement extends Node {
                expression:Node;

                constructor(loc:any, raw:string, cform:string, expression:Node) {
                    super("ExpressionStatement", loc, raw, cform);
                    this.expression = fromCena(expression);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ExpressionStatement(o.loc, o.raw, o.cform, o.expression);
                }

                toEsprima_():esprima.Syntax.ExpressionStatement {
                    return {
                        type: "ExpressionStatement",
                        loc: this.loc,
                        raw: this.raw, cform: this.cform,
                        expression: lib.utils.castTo<esprima.Syntax.Expression>(
                            this.expression.toEsprima()
                        )
                    }
                }

                toCString():string {
                    return this.expression.toCString();
                }

                children_():Node[] {
                    return [this.expression];
                }

                hasChildren_():boolean {
                    return !(this.expression instanceof EmptyExpression);
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.postOrderTraverse(visit, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.preOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    return this.expression.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.expression.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.expression.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ErrorNode extends Node {
                constructor(raw?:string) {
                    super("ErrorNode", unknownLocation(), raw, raw);
                }
            }


            export class SubscriptExpression extends Node {
                object:Node
                property:Node

                constructor(loc:any, raw:string, cform:string, object:any, property:any) {
                    super("SubscriptExpression", loc, raw, cform);
                    this.object = fromCena(object);
                    this.property = fromCena(property);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new SubscriptExpression(o.loc, o.raw, o.cform, o.object, o.property);
                }

                toCString():string {
                    return this.object.toCString() + "[" + this.property.toCString() + "]";
                }

                toEsprima_():esprima.Syntax.Node {
                  var self = this;
                  var loc = this.loc;
                  var sloc = builder.sourceLocation(
                    builder.position(loc.start.line, loc.start.column),
                    builder.position(loc.end.line, loc.end.column)
                    );
                    var libc = builder.memberExpression(
                      builder.identifier(
                        "lib",
                        sloc
                        ),
                        builder.identifier(
                          "c",
                          sloc
                          ),
                          false,
                          sloc
                          );

                  return builder.callExpression(
                    builder.memberExpression(libc, builder.identifier("getElement", sloc), false, sloc),
                    [
                      self.object.toEsprima(),
                      self.property.toEsprima()
                    ],
                    sloc
                    )
                    /*
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.Expression>(this.object.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: true,
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }*/
                }

                children_():Node[] {
                    return [this.object, this.property];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.object.postOrderTraverse(visit, data);
                    this.property.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.object.postOrderTraverse(visit, data);
                    return this.property.postOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.object.inOrderTraverse(visit, data);
                    return this.property.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.property.reversePostOrderTraverse(visit, data);
                    this.object.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.property.reversePreOrderTraverse(visit, data);
                    return this.object.reversePreOrderTraverse(visit, data);
                }
            }

            export class MemberExpression extends Node {
                right:Node
                left:Node
                operator:string
                computed:boolean

                constructor(loc:any, raw:string, cform:string, left:any, operator:string, right:any, computed?:boolean) {
                    super("MemberExpression", loc, raw, cform);
                    this.left = fromCena(left);
                    this.right = fromCena(right);
                    this.operator = operator;
                    this.computed = computed;
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new MemberExpression(o.loc, o.raw, o.cform, o.left, o.operator, o.right, o.computed);
                }

                toEsprima_():esprima.Syntax.MemberExpression {
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.Expression>(this.left.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.right.toEsprima()),
                        computed: this.computed,
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                toCString():string {
                    if (this.computed === true || isUndefined(this.computed)) {
                        return this.left.toCString() + this.operator + this.right.toCString();
                    } else {
                        return this.left.toCString() + "[" + this.right.toCString() + "]";
                    }
                }

                children_():Node[] {
                    return [this.left, this.right];
                }

                hasChildren_():boolean {
                    return true;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.postOrderTraverse(visit, data);
                    this.right.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.left.postOrderTraverse(visit, data);
                    return this.right.postOrderTraverse(visit, data);
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.left.inOrderTraverse(visit, data);
                    visit(this, data);
                    return this.right.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    this.right.reversePostOrderTraverse(visit, data);
                    this.left.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    visit(this, data);
                    this.right.reversePreOrderTraverse(visit, data);
                    return this.left.reversePreOrderTraverse(visit, data);
                }
            }

            export class ArrayExpression extends Node {
                elements:Node[]

                constructor(loc:any, raw:string, cform:string, elements:any[]) {
                    super("ArrayExpression", loc, raw, cform);
                    this.elements = _.map(elements, fromCena);
                    this.setChildParents();
                }

                static fromCena(o:any):Node {
                    return new ArrayExpression(o.loc, o.raw, o.cform, o.elements);
                }

                toEsprima_():esprima.Syntax.ArrayExpression {
                    return {
                        type: "ArrayExpression",
                        elements: castTo<esprima.Syntax.Expression[]>(_.map(this.elements, (elem) => elem.toEsprima())),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                toCString():string {
                    return "[" + _.map(this.elements, (elem:Node) => elem.toCString()).join(", ") + "]";
                }

                children_():Node[] {
                    return this.elements;
                }

                hasChildren_():boolean {
                    return this.elements.length > 0;
                }

                postOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    _.each(this.elements, (elem) => elem.postOrderTraverse(visit, data));
                    return visit(this, data);
                }

                preOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    var res:Node;
                    visit(this, data);
                    _.each(this.elements, (elem) => res = elem.preOrderTraverse(visit, data));
                    return res;
                }

                inOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    _.each(this.elements, (elem) => elem.inOrderTraverse(visit, data));
                    return visit(this, data);
                }

                reversePostOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    _.each(this.elements, (elem) => elem.reversePostOrderTraverse(visit, data));
                    return visit(this, data);
                }

                reversePreOrderTraverse_(visit:(Node, any) => Node, data:any):Node {
                    var res:Node;
                    visit(this, data);
                    _.each(this.elements, (elem) => res = elem.reversePreOrderTraverse(visit, data));
                    return res;
                }
            }

            var dispatch:Map<string, (o:any) => Node> = new Map<string, (o:any) => Node>();

            export function fromCena(o:any):Node {
                if (isUndefined(o) || isUndefined(o.type)) {
                    return new EmptyExpression();
                } else if (lib.utils.isArray(o)) {
                    return castTo<Node>(new CompoundNode(o));
                } else if (!dispatch.has(o.type)) {
                    lib.utils.logger.trace("Invalid input type toEsprima " + o.type);
                    return new ErrorNode(JSON.stringify(o));
                }
                var f = dispatch.get(o.type);
                return f(o);
            }

            export function toJS(o:any):{ code: string; map: lib.ast.sourcemap.SourceNode; } { // from Cena
                var nd:esprima.Syntax.Node = fromCena(o).toEsprima();
                return lib.ast.gen.generate(
                    nd,
                    // we might have to do  extra think here (see https://github.com/estools/escodegen/wiki/Source-Map-Usage )
                    {sourceMap: true, sourceMapWithCode: true, comment: true}
                );
            }

            var initialized:boolean = false;

            export function init() {
                if (initialized) {
                    return;
                }
                initialized = true;
                dispatch.set("EmptyExpression", EmptyExpression.fromCena);
                dispatch.set("NullNode", NullNode.fromCena);
                dispatch.set("StringLiteral", StringLiteral.fromCena);
                dispatch.set("BooleanLiteral", BooleanLiteral.fromCena);
                dispatch.set("CharLiteral", CharLiteral.fromCena);
                dispatch.set("Integer8Literal", Integer8Literal.fromCena);
                dispatch.set("Integer32Literal", Integer32Literal.fromCena);
                dispatch.set("Integer64Literal", Integer64Literal.fromCena);
                dispatch.set("Float32Literal", Float32Literal.fromCena);
                dispatch.set("Float64Literal", Float64Literal.fromCena);
                dispatch.set("TypeExpression", TypeExpression.fromCena);
                dispatch.set("TypeSpecification", TypeExpression.fromCena);
                dispatch.set("Identifier", Identifier.fromCena);
                dispatch.set("BlockStatement", BlockStatement.fromCena);
                dispatch.set("FunctionExpression", FunctionExpression.fromCena);
                dispatch.set("CallExpression", CallExpression.fromCena);
                dispatch.set("ParenExpression", ParenExpression.fromCena);
                dispatch.set("DereferenceExpression", DereferenceExpression.fromCena);
                dispatch.set("ReferenceExpression", ReferenceExpression.fromCena);
                dispatch.set("UnaryExpression", UnaryExpression.fromCena);
                dispatch.set("BinaryExpression", BinaryExpression.fromCena);
                dispatch.set("AssignmentExpression", AssignmentExpression.fromCena);
                dispatch.set("IfStatement", IfStatement.fromCena);
                dispatch.set("ConditionalExpression", ConditionalExpression.fromCena);
                dispatch.set("ForStatement", ForStatement.fromCena);
                dispatch.set("ProgramExpression", ProgramExpression.fromCena);
                dispatch.set("ReturnStatement", ReturnStatement.fromCena);
                dispatch.set("ExpressionStatement", ExpressionStatement.fromCena);
                dispatch.set("SubscriptExpression", SubscriptExpression.fromCena);
                dispatch.set("MemberExpression", MemberExpression.fromCena);
                dispatch.set("Program", ProgramExpression.fromCena);
                dispatch.set("Function", FunctionExpression.fromCena);
                dispatch.set("ParameterExpression", (o) => Identifier.fromCena(o.data));
                dispatch.set("VariableDeclaration", VariableDeclaration.fromCena);
                dispatch.set("VariableDeclarator", VariableDeclarator.fromCena);
                dispatch.set("SymbolLiteral", SymbolLiteral.fromCena);
                dispatch.set("Literal", SymbolLiteral.fromCena);
                dispatch.set("ReferenceType", ReferenceType.fromCena);
                dispatch.set("ArrayExpression", ArrayExpression.fromCena);
                //dispatch.set("WhileExpression", WhileExpression.fromCena);
            }

            init();

        }
    }
}
