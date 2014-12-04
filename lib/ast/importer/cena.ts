module lib.ast {
    export module importer {
        export module cena {
            import esprima = lib.ast.esprima;
            import castTo = lib.utils.castTo;
            import builder = lib.ast.types.builders;
            import isUndefined = lib.utils.isUndefined;

            var unknownLocation: esprima.Syntax.LineLocation = {
                start: {
                    line: -1,
                    column: -1
                },
                end: {
                    line: -1,
                    column: -1
                }
            };

            export class Node {
                type: string
                loc: esprima.Syntax.LineLocation
                raw: string
                cform: string

                constructor(type: string, loc: any, raw: string, cform: string) {
                    this.type = type;
                    this.loc = castTo<esprima.Syntax.LineLocation>(loc);
                    this.raw = raw;
                    this.cform = cform;
                }

                static fromCena(o: any): Node {
                    return new Node("Unknown", unknownLocation, "", "");
                }

                toEsprima(): esprima.Syntax.Node {
                    return {
                        type: "Comment",
                        value: "Node"
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class EmptyExpression extends Node {
                constructor() {
                    super("EmptyExpression", unknownLocation, "", "");
                }

                toEsprima() {
                    return {
                        type: "Comment",
                        value: "EmptyExpression"
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class NullNode extends Node {
                constructor() {
                    super("NullNode", unknownLocation, "", "");
                }

                toEsprima(): esprima.Syntax.Node {
                    return null;
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Literal<T> extends Node {
                value: T;

                constructor(loc: any, raw: string, cform: string, value: T) {
                    super("Literal", loc, raw, cform);
                    this.value = value;
                }

                toEsprima(): esprima.Syntax.Literal {
                    return {
                        type: "Literal",
                        value: this.value,
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class StringLiteral extends Literal<string> {
                constructor(loc: any, raw: string, cform: string, value: string) {
                    super(loc, raw, cform, value);
                    this.type = "StringLiteral";
                }

                static fromCena(o: any): StringLiteral {
                    return new StringLiteral(o.loc, o.raw, o.cform, o.value);
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class BooleanLiteral extends Literal<boolean> {
                constructor(loc: any, raw: string, cform: string, value: boolean) {
                    super(loc, raw, cform, value);
                    this.type = "BooleanLiteral";
                }

                static fromCena(o: any): Node {
                    return new BooleanLiteral(o.loc, o.raw, o.cform, o.value);
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class CharLiteral extends Node {
                value: string

                constructor(loc: any, raw: string, cform: string, value: string) {
                    super("CharLiteral", loc, raw, cform);
                    this.value = value;
                }

                static fromCena(o: any): CharLiteral {
                    return new CharLiteral(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "CharLiteral",
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Integer8Literal extends Node {
                value: number

                constructor(loc: any, raw: string, cform: string, value: number) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value
                }

                static fromCena(o: any): Integer8Literal {
                    return new Integer8Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int8",
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Integer32Literal extends Node {
                value: number

                constructor(loc: any, raw: string, cform: string, value: number) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value
                }

                static fromCena(o: any): Integer32Literal {
                    return new Integer32Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int32",
                            raw: this.raw, cform: this.cform,
                            loc: this.loc
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Integer64Literal extends Node {
                value: string

                constructor(loc: any, raw: string, cform: string, value: string) {
                    super("Integer8Literal", loc, raw, cform);
                    this.value = value
                }

                static fromCena(o: any): Integer64Literal {
                    return new Integer64Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int64",
                            loc: this.loc
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.loc,
                                raw: this.raw, cform: this.cform,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Float32Literal extends Node {
                value: number

                constructor(loc: any, raw: string, cform: string, value: number) {
                    super("FloatLiteral", loc, raw, cform);
                    this.value = value
                }

                static fromCena(o: any): Float32Literal {
                    return new Float32Literal(o.loc, o.raw, o.cform, o.value);
                }

                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.loc,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Float32",
                            loc: this.loc
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.loc,
                                value: this.value
                            })
                        ]
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Float64Literal extends Literal<number> {
                constructor(loc: any, raw: string, cform: string, value: number) {
                    super(loc, raw, cform, value);
                    this.type = "BooleanLiteral";
                }

                static fromCena(o: any): Node {
                    return new Float64Literal(o.loc, o.raw, o.cform, o.value);
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class TypeExpression extends Node {
                addressSpace: string[]
                qualifiers: string[]
                bases: string[]

                constructor(loc: any, raw: string, cform: string, addressSpace: string[], qualifiers: string[], bases: string[]) {
                    super("TypeExpression", loc, raw, cform);
                    this.addressSpace = addressSpace;
                    this.qualifiers = qualifiers;
                    this.bases = bases;
                }

                static fromCena(o: any): TypeExpression {
                    return new TypeExpression(o.loc, o.raw, o.cform, o.addressSpace, o.qualifiers, o.bases);
                }

                toEsprima(): esprima.Syntax.Comment {
                    return {
                        type: "Comment",
                        value: [this.addressSpace, this.qualifiers, this.bases].join(" "),
                        raw: JSON.stringify({
                            addressSpace: this.addressSpace,
                            qualifiers: this.qualifiers,
                            bases: this.bases
                        }),
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class Identifier extends Node {
                kind: TypeExpression
                name: string

                constructor(loc: any, raw: string, cform: string, name: string, kind?: any) {
                    super("Identifier", loc, raw, cform);
                    this.name = name;
                    if (isUndefined(kind)) {
                        this.kind = castTo<TypeExpression>(new EmptyExpression());
                    } else {
                        this.kind = TypeExpression.fromCena(kind);
                    }
                    ;
                }

                static fromCena(o: any): Identifier {
                    return new Identifier(o.loc, o.raw, o.cform, o.name, o.kind);
                }

                toEsprima(): esprima.Syntax.Identifier {
                    return {
                        type: "Identifier",
                        name: this.name,
                        kind: this.kind.toEsprima(),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    return visit(this, data);
                }
            }
            export class CompoundNode {
                elements: Node[]

                constructor(elements: any[]) {
                    this.elements = isUndefined(elements) ? [] : elements.map((elem) => fromCena(elem));
                }

                static fromCena(o: any): CompoundNode {
                    return new CompoundNode(o);
                }

                toEsprima(): esprima.Syntax.Node[] {
                    if (isUndefined(this.elements)) {
                        return [];
                    } else {
                        return this.elements.map((elem) => elem.toEsprima());
                    }
                }

                hasChildren(): boolean {
                    return _.isEmpty(this.elements);
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res: Node;
                    _.each(this.elements, (elem) => res = elem.postOrderTraverse(visit, data));
                    return res;
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res: Node;
                    _.each(this.elements, (elem) => res = elem.preOrderTraverse(visit, data));
                    return res;
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res: Node;
                    _.each(this.elements, (elem) => res = elem.inOrderTraverse(visit, data));
                    return res;
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res: Node;
                    _.forEachRight(this.elements, (elem) => res = elem.reversePostOrderTraverse(visit, data));
                    return res;
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res: Node;
                    _.forEachRight(this.elements, (elem) => res = elem.reversePreOrderTraverse(visit, data));
                    return res;
                }
            }
            export class BlockStatement extends Node {
                body: CompoundNode

                constructor(loc: any, raw: string, cform: string, body: any) {
                    super("BlockStatement", loc, raw, cform);
                    this.body = new CompoundNode(body.body);
                }

                static fromCena(o: any): BlockStatement {
                    return new BlockStatement(o.loc, o.raw, o.cform, o.body);
                }

                toEsprima(): esprima.Syntax.BlockStatement {
                    return {
                        type: "BlockStatement",
                        body: castTo<esprima.Syntax.SomeStatement[]>(this.body.toEsprima()),
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                hasChildren(): boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class FunctionExpression extends Node {
                attributes: string[]
                ret: Node
                id: Identifier
                params: CompoundNode
                body: BlockStatement

                constructor(loc: any, raw: string, cform: string, attributes: string[], ret: any, id: any, params: any[], body: any) {
                    super("FunctionExpression", loc, raw, cform);
                    this.attributes = attributes;
                    this.ret = isUndefined(ret) ? new EmptyExpression() : TypeExpression.fromCena(ret);
                    this.id = Identifier.fromCena(id);
                    this.params = CompoundNode.fromCena(params);
                    this.body = BlockStatement.fromCena({ loc: loc, raw: raw, cform: cform, body: body });
                }

                static fromCena(o: any): Node {
                    return new FunctionExpression(o.loc, o.raw, o.cform, o.attributes, o.ret, o.id, o.params, o.body);
                }

                toEsprima(): esprima.Syntax.Function {
                    return {
                        type: "FunctionExpression",
                        id: this.id.toEsprima(),
                        params: this.params.toEsprima(),
                        body: castTo<esprima.Syntax.BlockStatementOrExpression>(this.body.toEsprima()),
                        ret: this.ret.toEsprima(),
                        attributes: this.attributes,
                        defaults: [],
                        generator: false,
                        expression: false,
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.ret.postOrderTraverse(visit, data);
                    this.id.postOrderTraverse(visit, data);
                    this.params.postOrderTraverse(visit, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.ret.preOrderTraverse(visit, data);
                    this.id.preOrderTraverse(visit, data);
                    this.params.preOrderTraverse(visit, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.ret.inOrderTraverse(visit, data);
                    this.id.inOrderTraverse(visit, data);
                    this.params.inOrderTraverse(visit, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    this.params.reversePostOrderTraverse(visit, data);
                    this.id.reversePostOrderTraverse(visit, data);
                    this.ret.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    this.params.reversePreOrderTraverse(visit, data);
                    this.id.reversePreOrderTraverse(visit, data);
                    this.ret.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class CallExpression extends Node {
                callee: Identifier
                args: CompoundNode
                config: Node
                isCUDA: boolean = false

                constructor(loc: any, raw: string, cform: string, callee: any, args: any[], config?: any) {
                    super("CallExpression", loc, raw, cform);
                    this.callee = Identifier.fromCena(callee);
                    this.args = new CompoundNode(args);
                    this.config = isUndefined(config) ? new EmptyExpression() : fromCena(config);
                    this.isCUDA = !isUndefined(config);
                }

                static fromCena(o: any): Node {
                    return new CallExpression(o.loc, o.raw, o.cform, o.callee, castTo<any[]>(o.args), o.config);
                }

                toEsprima(): esprima.Syntax.CallExpression {
                    return {
                        type: "CallExpression",
                        config: this.config.toEsprima(),
                        isCUDA: this.isCUDA,
                        callee: castTo<esprima.Syntax.SomeExpression>(this.callee.toEsprima()),
                        arguments: this.args.toEsprima(),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return false;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.callee.postOrderTraverse(visit, data);
                    this.config.postOrderTraverse(visit, data);
                    return this.args.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.callee.preOrderTraverse(visit, data);
                    this.config.postOrderTraverse(visit, data);
                    return this.args.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.callee.inOrderTraverse(visit, data);
                    this.config.postOrderTraverse(visit, data);
                    return this.args.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.args.reversePostOrderTraverse(visit, data);
                    this.config.postOrderTraverse(visit, data);
                    this.callee.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.args.reversePreOrderTraverse(visit, data);
                    this.config.postOrderTraverse(visit, data);
                    this.callee.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ParenExpression extends Node {
                expression: Node

                constructor(loc: any, raw: string, cform: string, expression: any) {
                    super("ParenExpression", loc, raw, cform);
                    this.expression = fromCena(expression);
                }

                static fromCena(o: any): Node {
                    return new ParenExpression(o.loc, o.raw, o.cform, o.expression);
                }

                toEsprima(): esprima.Syntax.ExpressionStatement {
                    return {
                        type: "ExpressionStatement",
                        expression: castTo<esprima.Syntax.SomeExpression>(this.expression.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.expression.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.expression.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class DereferenceExpression extends Node {
                argument: Node
                rawArgument: any

                constructor(loc: any, raw: string, cform: string, argument: any) {
                    super("DereferenceExpression", loc, raw, cform);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                }

                static fromCena(o: any): Node {
                    return new DereferenceExpression(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.loc, this.raw, this.cform, new Identifier(this.loc, this.raw, this.cform, "dereference"), [this.rawArgument]);
                    return call.toEsprima();
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ReferenceExpression extends Node {
                argument: Node
                rawArgument: any

                constructor(loc: any, raw: string, cform: string, argument: any) {
                    super("ReferenceExpression", loc, raw, cform);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                }

                static fromCena(o: any): Node {
                    return new ReferenceExpression(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.loc, this.raw, this.cform, new Identifier(this.loc, this.raw, this.cform, "reference"), [this.rawArgument]);
                    return call.toEsprima();
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class UnaryExpression extends Node {
                operator: string
                rawArgument: any
                argument: Node

                constructor(loc: any, raw: string, cform: string, operator: string, argument: any) {
                    super("UnaryExpression", loc, raw, cform);
                    this.operator = operator
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                }

                static fromCena(o: any): Node {
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

                toEsprima(): esprima.Syntax.CallExpression;
                toEsprima(): esprima.Syntax.UnaryExpression;
                toEsprima(): any {
                    if (this.operator === "*") {
                        var nd = new DereferenceExpression(this.loc, this.raw, this.cform, this.rawArgument);
                        return nd.toEsprima();
                    } else if (this.operator === "&") {
                        var nd = new ReferenceExpression(this.loc, this.raw, this.cform, this.rawArgument);
                        return nd.toEsprima();
                    }
                    return {
                        type: "UnaryExpression",
                        argument: this.argument.toEsprima(),
                        operator: this.operator,
                        loc: this.loc,
                        raw: this.raw, cform: this.cform
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            function makeOp(op: string, symbol: string): Identifier {
                return new Identifier(unknownLocation, op, op, symbol);
            }

            export class BinaryExpression extends Node {
                operator: string
                right: Node
                left: Node
                static PropertyTable: {
                    [key: string]: Identifier;
                } = {
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

                constructor(loc: any, raw: string, cform: string, operator: string, right: any, left: any) {
                    super("BinaryExpression", loc, raw, cform);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                }

                static fromCena(o: any): Node {
                    return new BinaryExpression(o.loc, o.raw, o.cform, o.operator, o.left, o.right);
                }

                get property(): Identifier {
                    return BinaryExpression.PropertyTable[this.operator];
                }

                toEsprima(): esprima.Syntax.CallExpression {
                    var method: esprima.Syntax.MemberExpression = {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.SomeExpression>(this.right.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: false, /* A member expression. If computed === true,
                         the node corresponds to a computed e1[e2] expression and property is an Expression.
                         If computed === false, the node corresponds to a static e1.x expression and
                         property is an Identifier.*/
                        loc: this.right.loc,
                        raw: this.right.raw, cform: this.right.cform,
                    };
                    return {
                        type: "CallExpression",
                        callee: castTo<esprima.Syntax.SomeExpression>(method),
                        arguments: castTo<esprima.Syntax.SomeExpression[]>([this.left.toEsprima()]),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.left.postOrderTraverse(visit, data);
                    this.right.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.left.preOrderTraverse(visit, data);
                    return this.right.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.left.inOrderTraverse(visit, data);
                    visit(this, data);
                    return this.right.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.left.reversePostOrderTraverse(visit, data);
                    return this.right.reversePostOrderTraverse(visit, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.right.reversePreOrderTraverse(visit, data);
                    this.left.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class VariableDeclarator extends Node {
                init: Node
                id: Identifier

                constructor(loc: any, raw: string, cform: string, init: any, id: any) {
                    super("VariableDeclarator", loc, raw, cform);
                    this.init = isUndefined(init) ? new EmptyExpression() : fromCena(init);
                    this.id = Identifier.fromCena(id);
                }

                static fromCena(o: any): Node {
                    return new VariableDeclarator(o.loc, o.raw, o.cform, o.init, o.id);
                }

                toEsprima(): esprima.Syntax.VariableDeclarator {
                    return {
                        type: "VariableDeclarator",
                        init: castTo<esprima.Syntax.SomeExpression>(this.init.toEsprima()),
                        id: this.id.toEsprima(),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }
                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    lib.utils.logger.fatal("unimplemented");
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    lib.utils.logger.fatal("unimplemented");
                    return visit(this, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    lib.utils.logger.fatal("unimplemented");
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    lib.utils.logger.fatal("unimplemented");
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    lib.utils.logger.fatal("unimplemented");
                    return visit(this, data);
                }
            }

            export class VariableDeclaration extends Node {
                declarations: Node[]

                constructor(loc: any, raw: string, cform: string, declarations: any) {
                    super("VariableDeclaration", loc, raw, cform);
                    this.declarations = declarations.map(fromCena);
                }

                static fromCena(o: any): Node {
                    return new VariableDeclaration(o.loc, o.raw, o.cform, o.declarations);
                }

                toEsprima(): esprima.Syntax.VariableDeclaration {
                    return {
                        type: "VariableDeclaration",
                        declarations: castTo<esprima.Syntax.VariableDeclarator[]>(this.declarations.map((decl) => decl.toEsprima())),
                        kind: "var",
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    _.each(this.declarations, (decl) => decl.postOrderTraverse(visit, data));
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res;
                    visit(this, data);
                    _.each(this.declarations, (decl) => res = decl.preOrderTraverse(visit, data));
                    return res;
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    _.each(this.declarations, (decl) => decl.inOrderTraverse(visit, data));
                    return visit(this, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res;
                    visit(this, data);
                    _.eachRight(this.declarations, (decl) => res = decl.reversePostOrderTraverse(visit, data));
                    return res;
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    var res;
                    _.eachRight(this.declarations, (decl) => res = decl.reversePreOrderTraverse(visit, data));
                    return visit(this, data);
                }
            }
            export class AssignmentExpression extends Node {
                operator: string
                right: Node
                left: Node

                constructor(loc: any, raw: string, cform: string, operator: string, right: any, left: any) {
                    super("AssignmentExpression", loc, raw, cform);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                }

                static fromCena(o: any): Node {
                    return new AssignmentExpression(o.loc, o.raw, o.cform, o.operator, o.left, o.right);
                }

                toEsprima(): esprima.Syntax.AssignmentExpression {
                    return {
                        type: "AssignmentExpression",
                        operator: this.operator,
                        left: castTo<esprima.Syntax.SomeExpression>(this.left.toEsprima()),
                        right: castTo<esprima.Syntax.SomeExpression>(this.right.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.left.postOrderTraverse(visit, data);
                    this.right.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.left.preOrderTraverse(visit, data);
                    return this.right.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.left.inOrderTraverse(visit, data);
                    visit(this, data);
                    return this.right.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.left.reversePostOrderTraverse(visit, data);
                    return this.right.reversePostOrderTraverse(visit, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.right.reversePreOrderTraverse(visit, data);
                    this.left.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class IfStatement extends Node {
                test: Node
                consequent: Node
                alternate: Node

                constructor(loc: any, raw: string, cform: string, test: any, consequent: any, alternate?: any) {
                    super("IfStatement", loc, raw, cform);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = isUndefined(alternate) ? new EmptyExpression() : fromCena(alternate);
                }

                static fromCena(o: any): Node {
                    return new IfStatement(o.loc, o.raw, o.cform, o.test, o.consequent, o.alternate);
                }

                toEsprima(): esprima.Syntax.IfStatement {
                    return {
                        type: "IfStatement",
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.SomeStatement>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.SomeStatement>(this.consequent.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.test.postOrderTraverse(visit, data);
                    this.alternate.postOrderTraverse(visit, data);
                    this.consequent.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.test.preOrderTraverse(visit, data);
                    this.alternate.preOrderTraverse(visit, data);
                    return this.consequent.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.test.inOrderTraverse(visit, data);
                    this.alternate.inOrderTraverse(visit, data);
                    return this.consequent.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.consequent.reversePostOrderTraverse(visit, data);
                    this.alternate.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.consequent.reversePreOrderTraverse(visit, data);
                    this.alternate.reversePreOrderTraverse(visit, data);
                    return this.test.reversePreOrderTraverse(visit, data);
                }
            }
            export class ConditionalExpression extends Node {
                test: Node
                consequent: Node
                alternate: Node

                constructor(loc: any, raw: string, cform: string, test: any, consequent: any, alternate?: any) {
                    super("ConditionalExpression", loc, raw, cform);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = fromCena(alternate);
                }

                static fromCena(o: any): Node {
                    return new ConditionalExpression(o.loc, o.raw, o.cform, o.test, o.consequent, o.alternate);
                }

                toEsprima(): esprima.Syntax.ConditionalExpression {
                    return {
                        type: "ConditionalExpression",
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.SomeExpression>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.SomeExpression>(this.consequent.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.test.postOrderTraverse(visit, data);
                    this.alternate.postOrderTraverse(visit, data);
                    this.consequent.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.test.preOrderTraverse(visit, data);
                    this.alternate.preOrderTraverse(visit, data);
                    return this.consequent.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.test.inOrderTraverse(visit, data);
                    this.alternate.inOrderTraverse(visit, data);
                    return this.consequent.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.consequent.reversePostOrderTraverse(visit, data);
                    this.alternate.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.consequent.reversePreOrderTraverse(visit, data);
                    this.alternate.reversePreOrderTraverse(visit, data);
                    return this.test.reversePreOrderTraverse(visit, data);
                }
            }
            export class ForStatement extends Node {
                init: Node
                test: Node
                update: Node
                body: Node

                constructor(loc: any, raw: string, cform: string, init: any, test: any, update: any, body: any) {
                    super("ForStatement", loc, raw, cform);
                    this.init = fromCena(init);
                    this.test = fromCena(test);
                    this.update = fromCena(update);
                    this.body = fromCena(body);
                }

                static fromCena(o: any): Node {
                    return new ForStatement(o.loc, o.raw, o.cform, o.init, o.test, o.update, o.body);
                }

                toEsprima(): esprima.Syntax.ForStatement {
                    return {
                        type: "ForStatement",
                        init: castTo<esprima.Syntax.VariableDeclaratorOrExpression>(this.init.toEsprima()),
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        update: castTo<esprima.Syntax.SomeExpression>(this.update.toEsprima()),
                        body: castTo<esprima.Syntax.SomeStatement>(this.body.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.init.postOrderTraverse(visit, data);
                    this.test.postOrderTraverse(visit, data);
                    this.update.postOrderTraverse(visit, data);
                    this.body.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.init.preOrderTraverse(visit, data);
                    this.test.preOrderTraverse(visit, data);
                    this.update.preOrderTraverse(visit, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.init.inOrderTraverse(visit, data);
                    this.test.inOrderTraverse(visit, data);
                    this.update.inOrderTraverse(visit, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    this.update.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    this.init.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.body.reversePostOrderTraverse(visit, data);
                    this.update.reversePostOrderTraverse(visit, data);
                    this.test.reversePostOrderTraverse(visit, data);
                    return this.init.reversePostOrderTraverse(visit, data);
                }

            }
            export class ProgramExpression extends Node {
                body: CompoundNode;

                constructor(loc: any, raw: string, cform: string, body: any[]) {
                    super("ProgramExpression", loc, raw, cform);
                    this.body = new CompoundNode(body);
                }

                static fromCena(o: any): Node {
                    return new ProgramExpression(o.loc, o.raw, o.cform, o.body);
                }

                toEsprima(): esprima.Syntax.Program {
                    return {
                        type: "Program",
                        body: castTo<esprima.Syntax.SomeStatement[]>(this.body.toEsprima()),
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return this.body.hasChildren();
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.body.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.body.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ReturnStatement extends Node {
                argument: Node;

                constructor(loc: any, raw: string, cform: string, argument?: Node) {
                    super("ReturnStatement", loc, raw, cform);
                    if (argument) {
                        this.argument = argument;
                    } else {
                        this.argument = new EmptyExpression();
                    }
                }

                static fromCena(o: any): Node {
                    return new ReturnStatement(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima(): esprima.Syntax.ReturnStatement {
                    return {
                        type: "ReturnStatement",
                        loc: this.loc,
                        raw: this.raw, cform: this.cform,
                        argument: lib.utils.castTo<esprima.Syntax.SomeExpression>(
                            this.argument.toEsprima()
                            )
                    }
                }

                hasChildren(): boolean {
                    return !(this.argument instanceof EmptyExpression);
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.argument.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.argument.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ExpressionStatement extends Node {
                expression: Node;

                constructor(loc: any, raw: string, cform: string, expression: Node) {
                    super("ExpressionStatement", loc, raw, cform);
                    this.expression = expression;
                }

                static fromCena(o: any): Node {
                    return new ReturnStatement(o.loc, o.raw, o.cform, o.argument);
                }

                toEsprima(): esprima.Syntax.ExpressionStatement {
                    return {
                        type: "ExpressionStatement",
                        loc: this.loc,
                        raw: this.raw, cform: this.cform,
                        expression: lib.utils.castTo<esprima.Syntax.SomeExpression>(
                            this.expression.toEsprima()
                            )
                    }
                }

                hasChildren(): boolean {
                    return !(this.expression instanceof EmptyExpression);
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.postOrderTraverse(visit, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.preOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    return this.expression.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.expression.reversePreOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.expression.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }
            }
            export class ErrorNode extends Node {
                constructor(raw?: string) {
                    super("ErrorNode", unknownLocation, raw, raw);
                }
            }


            export class SubscriptExpression extends Node {
                object: Node
                property: Node

                constructor(loc: any, raw: string, cform: string, object: any, property: any) {
                    super("SubscriptExpression", loc, raw, cform);
                    this.object = fromCena(init);
                    this.property = fromCena(property);
                }

                static fromCena(o: any): Node {
                    return new SubscriptExpression(o.loc, o.raw, o.cform, o.object, o.property);
                }

                toEsprima(): esprima.Syntax.MemberExpression {
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.SomeExpression>(this.object.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: true,
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.object.postOrderTraverse(visit, data);
                    this.property.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.object.postOrderTraverse(visit, data);
                    return this.property.postOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.object.inOrderTraverse(visit, data);
                    return this.property.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.property.reversePostOrderTraverse(visit, data);
                    this.object.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.property.reversePreOrderTraverse(visit, data);
                    return this.object.reversePreOrderTraverse(visit, data);
                }
            }

            export class MemberExpression extends Node {
                object: Node
                property: Node
                computed: boolean

                constructor(loc: any, raw: string, cform: string, object: any, property: any, computed: boolean) {
                    super("MemberExpression", loc, raw, cform);
                    this.object = fromCena(object);
                    this.property = fromCena(property);
                    this.computed = computed;
                }

                static fromCena(o: any): Node {
                    return new MemberExpression(o.loc, o.raw, o.cform, o.object, o.property, o.computed);
                }

                toEsprima(): esprima.Syntax.MemberExpression {
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.SomeExpression>(this.object.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: this.computed,
                        raw: this.raw, cform: this.cform,
                        loc: this.loc
                    }
                }

                hasChildren(): boolean {
                    return true;
                }

                postOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.object.postOrderTraverse(visit, data);
                    this.property.postOrderTraverse(visit, data);
                    return visit(this, data);
                }

                preOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.object.postOrderTraverse(visit, data);
                    return this.property.postOrderTraverse(visit, data);
                }

                inOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.object.inOrderTraverse(visit, data);
                    return this.property.inOrderTraverse(visit, data);
                }

                reversePostOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    this.property.reversePostOrderTraverse(visit, data);
                    this.object.reversePostOrderTraverse(visit, data);
                    return visit(this, data);
                }

                reversePreOrderTraverse(visit: (Node, any) => Node, data: any): Node {
                    visit(this, data);
                    this.property.reversePreOrderTraverse(visit, data);
                    return this.object.reversePreOrderTraverse(visit, data);
                }
            }


            var dispatch: Map<string, (o: any) => Node> = new Map<string, (o: any) => Node>();

            export function fromCena(o: any) {
                if (isUndefined(o) || isUndefined(o.type)) {
                    return new EmptyExpression();
                } else if (!dispatch.has(o.type)) {
                    lib.utils.logger.trace("Invalid input type toEsprima");
                    return new ErrorNode(JSON.stringify(o));
                }
                var f = dispatch.get(o.type);
                return f(o);
            }

            export function toJS(o: any): { code: string; map: lib.ast.sourcemap.SourceNode; } { // from Cena
                return lib.ast.gen.generate(
                    fromCena(o),
                    // we might have to do some extra think here (see https://github.com/estools/escodegen/wiki/Source-Map-Usage )
                    { sourceMap: true, sourceMapWithCode: true, comment: true }
                    );
            }

            var initialized: boolean = false;

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
            }

            init();

        }
    }
}
