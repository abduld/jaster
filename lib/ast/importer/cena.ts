

module lib.ast {
    export module importer {
        export module cena {
            import esprima = lib.ast.esprima;
            import castTo = lib.utils.castTo;
            import builder = lib.ast.types.builders;
            import isUndefined = lib.utils.isUndefined;

            export class Node {
                type: string;
                line: number;
                column: number;
                raw: string
                constructor(type: string, line: number, column: number, raw: string) {
                    this.type = type;
                    this.line = line;
                    this.column = column;
                    this.raw = raw
                }
                static fromCena(o: any): Node {
                    return new Node("Unknown", -1, -1, "");
                }
                toEsprima(): esprima.Syntax.Node {
                    return {
                        type: "Comment",
                        value: "Node"
                    }
                }
                get location(): esprima.Syntax.LineLocation {
                    return {
                        start: {
                            line: this.line,
                            column: this.column
                        },
                        end: {
                            line: this.line,
                            column: this.column + (isUndefined(this.raw) ? 0 : this.raw.length)
                        }
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
                    super("EmptyExpression", -1, -1, "");
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
                    super("NullNode", -1, -1, "");
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
                constructor(line: number, column: number, value: T, raw?: string) {
                    super("Literal", line, column, raw);
                    this.value = value;
                }
                toEsprima(): esprima.Syntax.Literal {
                    return {
                        type: "Literal",
                        value: this.value,
                        loc: this.location,
                        raw: this.raw
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
                constructor(line: number, column: number, value: string, raw?: string) {
                    super(line, column, value, raw);
                    this.type = "StringLiteral";
                }
                static fromCena(o: any): StringLiteral {
                    return new StringLiteral(o.line, o.column, o.value, o.raw);
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
                constructor(line: number, column: number, value: boolean, raw?: string) {
                    super(line, column, value, raw);
                    this.type = "BooleanLiteral";
                }
                static fromCena(o: any): Node {
                    return new BooleanLiteral(o.line, o.column, o.value, o.raw);
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
                constructor(line: number, column: number, value: string, raw?: string) {
                    super("CharLiteral", line, column, raw);
                    this.value = value;
                }
                static fromCena(o: any): CharLiteral {
                    return new CharLiteral(o.line, o.column, o.value, o.raw);
                }
                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.location,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "CharLiteral",
                            loc: this.location
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.location,
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
                constructor(line: number, column: number, value: number, raw?: string) {
                    super("Integer8Literal", line, column, raw);
                    this.value = value
                }
                static fromCena(o: any): Integer8Literal {
                    return new Integer8Literal(o.line, o.column, o.value, o.raw);
                }
                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.location,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int8",
                            loc: this.location
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.location,
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
                constructor(line: number, column: number, value: number, raw?: string) {
                    super("Integer8Literal", line, column, raw);
                    this.value = value
                }
                static fromCena(o: any): Integer32Literal {
                    return new Integer32Literal(o.line, o.column, o.value, o.raw);
                }
                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.location,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int32",
                            loc: this.location
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.location,
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
                constructor(line: number, column: number, value: string, raw?: string) {
                    super("Integer8Literal", line, column, raw);
                    this.value = value
                }
                static fromCena(o: any): Integer64Literal {
                    return new Integer64Literal(o.line, o.column, o.value, o.raw);
                }
                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.location,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Int64",
                            loc: this.location
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.location,
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
                constructor(line: number, column: number, value: number, raw?: string) {
                    super("FloatLiteral", line, column, raw);
                    this.value = value
                }
                static fromCena(o: any): Float32Literal {
                    return new Float32Literal(o.line, o.column, o.value, o.raw);
                }
                toEsprima(): esprima.Syntax.NewExpression {
                    return {
                        type: "NewExpression",
                        loc: this.location,
                        callee: castTo<esprima.Syntax.SomeExpression>({
                            type: "Identifier",
                            value: "Float32",
                            loc: this.location
                        }),
                        arguments: [
                            castTo<esprima.Syntax.SomeExpression>({
                                type: "Literal",
                                loc: this.location,
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
                constructor(line: number, column: number, value: number, raw?: string) {
                    super(line, column, value, raw);
                    this.type = "BooleanLiteral";
                }
                static fromCena(o: any): Node {
                    return new Float64Literal(o.line, o.column, o.value, o.raw);
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
                constructor(line: number, column: number, addressSpace: string[], qualifiers: string[], bases: string[], raw?: string) {
                    super("TypeExpression", line, column, raw);
                    this.addressSpace = addressSpace;
                    this.qualifiers = qualifiers;
                    this.bases = bases;
                }
                static fromCena(o: any): TypeExpression {
                    return new TypeExpression(o.line, o.column, o.addressSpace, o.qualifiers, o.bases, o.raw);
                }
                toEsprima(): esprima.Syntax.Comment {
                    return {
                        type: "Comment",
                        value: [this.addressSpace, this.qualifiers, this.bases].join(" "),
                        raw: JSON.stringify({ addressSpace: this.addressSpace, qualifiers: this.qualifiers, bases: this.bases }),
                        loc: this.location
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
                constructor(line: number, column: number, name: string, kind?: any, raw?: string) {
                    super("Identifier", line, column, raw);
                    this.name = name;
                    if (isUndefined(kind)) {
                        this.kind = castTo<TypeExpression>(new EmptyExpression());
                    } else {
                        this.kind = TypeExpression.fromCena(kind);
                    };
                }
                static fromCena(o: any): Identifier {
                    return new Identifier(o.line, o.column, o.name, o.kind, o.raw);
                }
                toEsprima(): esprima.Syntax.Identifier {
                    return {
                        type: "Identifier",
                        name: this.name,
                        kind: this.kind.toEsprima(),
                        raw: this.raw,
                        loc: this.location
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
                    this.elements = elements.map((elem) => fromCena(elem));
                }
                static fromCena(o: any): CompoundNode {
                    return new CompoundNode(o);
                }
                toEsprima(): esprima.Syntax.Node[] {
                    return this.elements.map((elem) => elem.toEsprima())
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

                constructor(line: number, column: number, body: any, raw?: string) {
                    super("BlockStatement", line, column, raw);
                    this.body = new CompoundNode(body.body);
                }
                static fromCena(o: any): BlockStatement {
                    return new BlockStatement(o.line, o.column, o.body, o.raw);
                }
                toEsprima(): esprima.Syntax.BlockStatement {
                    return {
                        type: "BlockStatement",
                        body: castTo<esprima.Syntax.SomeStatement[]>(this.body.toEsprima()),
                        loc: this.location,
                        raw: this.raw
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
                ret: TypeExpression
                id: Identifier
                params: CompoundNode
                body: BlockStatement
                constructor(line: number, column: number, attributes: string[], ret: any, id: any, params: any[], body: any, raw?: string) {
                    super("FunctionExpression", line, column, raw);
                    this.attributes = attributes;
                    this.ret = TypeExpression.fromCena(ret);
                    this.id = Identifier.fromCena(id);
                    this.params = CompoundNode.fromCena(params);
                    this.body = BlockStatement.fromCena(body);
                }
                static fromCena(o: any): Node {
                    return new FunctionExpression(o.line, o.column, o.attributes, o.ret, o.id, o.params, o.body, o.raw);
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
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, callee: any, args: any[], config? : any, raw?: string) {
                    super("CallExpression", line, column, raw);
                    this.callee = Identifier.fromCena(callee);
                    this.args = new CompoundNode(args);
                    this.config = fromCena(config);
                    this.isCUDA = !isUndefined(config);
                }
                static fromCena(o: any): Node {
                    return new CallExpression(o.line, o.column, o.callee, castTo<any[]>(o.args), o.config, o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    return {
                        type: "CallExpression",
                        config: this.config.toEsprima(),
                        isCUDA: this.isCUDA,
                        callee: castTo<esprima.Syntax.SomeExpression>(this.callee.toEsprima()),
                        arguments: this.args.toEsprima(),
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, expression: any, raw?: string) {
                    super("ParenExpression", line, column, raw);
                    this.expression = fromCena(expression);
                }
                static fromCena(o: any): Node {
                    return new CallExpression(o.line, o.column, o.expression, o.raw);
                }
                toEsprima(): esprima.Syntax.ExpressionStatement {
                    return {
                        type: "ExpressionStatement",
                        expression: castTo<esprima.Syntax.SomeExpression>(this.expression.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, argument: any, raw?: string) {
                    super("DereferenceExpression", line, column, raw);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                }
                static fromCena(o: any): Node {
                    return new DereferenceExpression(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.line, this.column, new Identifier(this.line, this.column, "dereference"), [this.rawArgument]);
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
                constructor(line: number, column: number, argument: any, raw?: string) {
                    super("ReferenceExpression", line, column, raw);
                    this.rawArgument = argument;
                    this.argument = fromCena(argument);
                }
                static fromCena(o: any): Node {
                    return new ReferenceExpression(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.line, this.column, new Identifier(this.line, this.column, "reference"), [this.rawArgument]);
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
                constructor(line: number, column: number, operator: string, argument: any, raw?: string) {
                    super("UnaryExpression", line, column, raw);
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
                        return new UnaryExpression(o.line, o.column, o.operator, o.argument, o.raw);
                    }
                }
                toEsprima(): esprima.Syntax.CallExpression;
                toEsprima(): esprima.Syntax.UnaryExpression;
                toEsprima(): any {
                    if (this.operator === "*") {
                        var nd = new DereferenceExpression(this.line, this.column, this.rawArgument, this.raw);
                        return nd.toEsprima();
                    } else if (this.operator === "&") {
                        var nd = new ReferenceExpression(this.line, this.column, this.rawArgument, this.raw);
                        return nd.toEsprima();
                    }
                    return {
                        type: "UnaryExpression",
                        argument: this.argument.toEsprima(),
                        operator: this.operator,
                        loc: this.location,
                        raw: this.raw
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
            export class BinaryExpression extends Node {
                operator: string
                right: Node
                left: Node
                static PropertyTable: {
                    [key: string]: Identifier;
                } = {
                    "+": new Identifier(0, 0, "plus", undefined, "+"),
                    "-": new Identifier(0, 0, "minus", undefined, "-"),
                    "*": new Identifier(0, 0, "times", undefined, "*"),
                    "/": new Identifier(0, 0, "divide", undefined, "/"),
                    "%": new Identifier(0, 0, "mod", undefined, "%"),
                    "&&": new Identifier(0, 0, "and", undefined, "&&"),
                    "||": new Identifier(0, 0, "or", undefined, "||"),
                    "==": new Identifier(0, 0, "equals", undefined, "=="),
                    "!=": new Identifier(0, 0, "notequals", undefined, "!="),
                    ">": new Identifier(0, 0, "greaterthan", undefined, ">"),
                    ">=": new Identifier(0, 0, "greaterthanequals", undefined, ">="),
                    "<": new Identifier(0, 0, "lessthan", undefined, "<"),
                    "<=": new Identifier(0, 0, "lessthanequals", undefined, "<="),
                    "|": new Identifier(0, 0, "bitor", undefined, "|"),
                    "&": new Identifier(0, 0, "bitand", undefined, "&"),
                    "^": new Identifier(0, 0, "bitxor", undefined, "^"),
                    ">>": new Identifier(0, 0, "shiftright", undefined, ">>"),
                    "<<": new Identifier(0, 0, "shiftleft", undefined, "<<")
                };
                constructor(line: number, column: number, operator: string, right: any, left: any, raw?: string) {
                    super("BinaryExpression", line, column, raw);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                }
                static fromCena(o: any): Node {
                    return new BinaryExpression(o.line, o.column, o.operator, o.left, o.right, o.raw);
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
                        loc: this.right.location,
                        raw: this.right.raw
                    };
                    return {
                        type: "CallExpression",
                        callee: castTo<esprima.Syntax.SomeExpression>(method),
                        arguments: castTo<esprima.Syntax.SomeExpression[]>([this.left.toEsprima()]),
                        raw: this.raw,
                        loc: this.location
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
            export class AssignmentExpression extends Node {
                operator: string
                right: Node
                left: Node
                constructor(line: number, column: number, operator: string, right: any, left: any, raw?: string) {
                    super("AssignmentExpression", line, column, raw);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                }
                static fromCena(o: any): Node {
                    return new AssignmentExpression(o.line, o.column, o.operator, o.left, o.right, o.raw);
                }
                toEsprima(): esprima.Syntax.AssignmentExpression {
                    return {
                        type: "AssignmentExpression",
                        operator: this.operator,
                        left: castTo<esprima.Syntax.SomeExpression>(this.left.toEsprima()),
                        right: castTo<esprima.Syntax.SomeExpression>(this.right.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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

                constructor(line: number, column: number, test: any, consequent: any, alternate?: any, raw?: string) {
                    super("IfStatement", line, column, raw);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = fromCena(alternate);
                }
                static fromCena(o: any): Node {
                    return new IfStatement(o.line, o.column, o.test, o.consequent, o.alternate, o.raw);
                }
                toEsprima(): esprima.Syntax.IfStatement {
                    return {
                        type: "IfStatement",
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.SomeStatement>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.SomeStatement>(this.consequent.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, test: any, consequent: any, alternate?: any, raw?: string) {
                    super("ConditionalExpression", line, column, raw);
                    this.test = fromCena(test);
                    this.consequent = fromCena(consequent);
                    this.alternate = fromCena(alternate);
                }
                static fromCena(o: any): Node {
                    return new ConditionalExpression(o.line, o.column, o.test, o.consequent, o.alternate, o.raw);
                }
                toEsprima(): esprima.Syntax.ConditionalExpression {
                    return {
                        type: "ConditionalExpression",
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        alternate: castTo<esprima.Syntax.SomeExpression>(this.alternate.toEsprima()),
                        consequent: castTo<esprima.Syntax.SomeExpression>(this.consequent.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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

                constructor(line: number, column: number, init: any, test: any, update: any, body: any, raw?: string) {
                    super("ForStatement", line, column, raw);
                    this.init = fromCena(init);
                    this.test = fromCena(test);
                    this.update = fromCena(update);
                    this.body = fromCena(body);
                }
                static fromCena(o: any): Node {
                    return new ForStatement(o.line, o.column, o.init, o.test, o.update, o.body, o.raw);
                }
                toEsprima(): esprima.Syntax.ForStatement {
                    return {
                        type: "ForStatement",
                        init: castTo<esprima.Syntax.VariableDeclaratorOrExpression>(this.init.toEsprima()),
                        test: castTo<esprima.Syntax.SomeExpression>(this.test.toEsprima()),
                        update: castTo<esprima.Syntax.SomeExpression>(this.update.toEsprima()),
                        body: castTo<esprima.Syntax.SomeStatement>(this.body.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, body: any[], raw?: string) {
                    super("ProgramExpression", line, column, raw);
                    this.body = new CompoundNode(body);
                }
                static fromCena(o: any): Node {
                    return new ProgramExpression(o.line, o.column, o.body, o.raw);
                }
                toEsprima(): esprima.Syntax.Program {
                    return {
                        type: "Program",
                        body: castTo<esprima.Syntax.SomeStatement[]>(this.body.toEsprima()),
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, argument?: Node, raw?: string) {
                    super("ReturnStatement", line, column, raw);
                    if (argument) {
                        this.argument = argument;
                    } else {
                        this.argument = new EmptyExpression();
                    }
                }
                static fromCena(o: any): Node {
                    return new ReturnStatement(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.ReturnStatement {
                    return {
                        type: "ReturnStatement",
                        loc: this.location,
                        raw: this.raw,
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
                constructor(line: number, column: number, expression: Node, raw?: string) {
                    super("ExpressionStatement", line, column, raw);
                    this.expression = expression;
                }
                static fromCena(o: any): Node {
                    return new ReturnStatement(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.ExpressionStatement {
                    return {
                        type: "ExpressionStatement",
                        loc: this.location,
                        raw: this.raw,
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
                    super("ErrorNode", -1, -1, raw);
                }
            }


            export class SubscriptExpression extends Node {
                object: Node
                property: Node
                constructor(line: number, column: number, object: any, property: any, raw?: string) {
                    super("SubscriptExpression", line, column, raw);
                    this.object = fromCena(init);
                    this.property = fromCena(property);
                }
                static fromCena(o: any): Node {
                    return new SubscriptExpression(o.line, o.column, o.object, o.property, o.raw);
                }
                toEsprima(): esprima.Syntax.MemberExpression {
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.SomeExpression>(this.object.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: true,
                        raw: this.raw,
                        loc: this.location
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
                constructor(line: number, column: number, object: any, property: any, computed: boolean, raw?: string) {
                    super("MemberExpression", line, column, raw);
                    this.object = fromCena(init);
                    this.property = fromCena(property);
                    this.computed = computed;
                }
                static fromCena(o: any): Node {
                    return new MemberExpression(o.line, o.column, o.object, o.property, o.computed, o.raw);
                }
                toEsprima(): esprima.Syntax.MemberExpression {
                    return {
                        type: "MemberExpression",
                        object: castTo<esprima.Syntax.SomeExpression>(this.object.toEsprima()),
                        property: castTo<esprima.Syntax.IdentifierOrExpression>(this.property.toEsprima()),
                        computed: this.computed,
                        raw: this.raw,
                        loc: this.location
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
                if (isUndefined(o) || isUndefined(o.type) || !dispatch.has(o.type)) {
                    lib.utils.logger.error("Invalid input type toEsprima");
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
            }

            init();

        }
    }
}
