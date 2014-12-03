

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
                            column: this.column
                        }
                    }
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
            }
            export class NullNode extends Node {
                constructor() {
                    super("NullNode", -1, -1, "");
                }
                toEsprima() {
                    return {
                        type: "Comment",
                        value: "NullNode"
                    }
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
            }

            export class StringLiteral extends Literal<string> {
                constructor(line: number, column: number, value: string, raw?: string) {
                    super(line, column, value, raw);
                    this.type = "StringLiteral";
                }
                static fromCena(o: any): StringLiteral {
                    return new StringLiteral(o.line, o.column, o.value, o.raw);
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
            }

            export class Float64Literal extends Literal<number> {
                constructor(line: number, column: number, value: number, raw?: string) {
                    super(line, column, value, raw);
                    this.type = "BooleanLiteral";
                }
                static fromCena(o: any): Node {
                    return new Float64Literal(o.line, o.column, o.value, o.raw);
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
            }

            export class CallExpression extends Node {
                callee: Identifier
                args: CompoundNode
                constructor(line: number, column: number, callee: any, args: any[], raw?: string) {
                    super("CallExpression", line, column, raw);
                    this.callee = Identifier.fromCena(callee);
                    this.args = new CompoundNode(args);
                }
                static fromCena(o: any): Node {
                    return new CallExpression(o.line, o.column, o.callee, castTo<any[]>(o.args), o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    return {
                        type: "CallExpression",
                        callee: castTo<esprima.Syntax.SomeExpression>(this.callee.toEsprima()),
                        arguments: this.args.toEsprima(),
                        raw: this.raw,
                        loc: this.location
                    }
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
            }

            export class DereferenceExpression extends Node {
                argument: any
                constructor(line: number, column: number, argument: any, raw?: string) {
                    super("DereferenceExpression", line, column, raw);
                    this.argument = argument;
                }
                static fromCena(o: any): Node {
                    return new DereferenceExpression(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.line, this.column, new Identifier(this.line, this.column, "dereference"), [this.argument]);
                    return call.toEsprima();
                }
            }

            export class ReferenceExpression extends Node {
                argument: any
                constructor(line: number, column: number, argument: any, raw?: string) {
                    super("ReferenceExpression", line, column, raw);
                    this.argument = argument;
                }
                static fromCena(o: any): Node {
                    return new ReferenceExpression(o.line, o.column, o.argument, o.raw);
                }
                toEsprima(): esprima.Syntax.CallExpression {
                    var call: CallExpression = new CallExpression(this.line, this.column, new Identifier(this.line, this.column, "reference"), [this.argument]);
                    return call.toEsprima();
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
                    super("UnaryExpression", line, column, raw);
                    this.operator = operator
                    this.right = fromCena(right);
                    this.left = fromCena(left);
                }
                static fromCena(o: any): Node {
                    return new BinaryExpression(o.line, o.column, o.operator, o.argument, o.raw);
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
            }

            export class IfStatement extends Node {

            }

            export class ForStatement extends Node {

            }

            export class ProgramExpression extends Node {
                body: Node[];
                constructor(line: number, column: number, body: Node[], raw?: string) {
                    super("ProgramExpression", line, column, raw);
                    this.body = body;
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
                toEsprima(): esprima.Syntax.ReturnStatement {
                    return {
                        type: "ReturnStatement",
                        loc: this.location,
                        argument: lib.utils.castTo<esprima.Syntax.SomeExpression>(
                            this.argument.toEsprima()
                            )
                    }
                }
            }

            export class ExpressionStatement extends Node {
                expression: Node;
                constructor(line: number, column: number, expression: Node, raw?: string) {
                    super("ExpressionStatement", line, column, raw);
                    this.expression = expression;
                }
            }

            export class ErrorNode extends Node {
                constructor(raw?: string) {
                    super("ErrorNode", -1, -1, raw);
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
                dispatch.set("test", Node.fromCena);
            }

            init();

        }
    }
}
