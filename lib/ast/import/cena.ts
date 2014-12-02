

module lib.ast {
  export module importer {
  export module cena {
        export class Node {
          type : string;
          line : number;
          column : number;
          constructor(type : string, line : number, column : number) {
            this.type = type;
            this.line = line;
            this.column = column;
          }
        }
        export class Literal<T> extends Node {
          value : T;
          constructor(line: number, column : number, value : T) {
            super("Literal", line, column);
            this.value = value;
          }
        }

        export class StringLiteral extends Literal<string> {
          constructor(line: number, column : number, value : string) {
            super(line, column, value);
            this.type = "StringLiteral";
          }
        }

        export class CharLiteral extends Literal<string> {
          constructor(line: number, column : number, value : string) {
            super(line, column, value);
            this.type = "CharLiteral";
          }
        }

        export class Integer8Literal extends Node {

        }

        export class Integer32Literal extends Node {

        }

        export class Integer64Literal extends Node {

        }

        export class TypeExpression extends Node {

        }

        export class FunctionExpression extends Nodes {

        }

        export class CallExpression extends Node {

        }

        export class ParenExpression extends Node {

        }

        export class DereferenceExpression extends Node {

        }

        export class ReferenceExpression extends Node {

        }

        export class UnaryExpression extends Node {

        }

        export class BinaryExpression extends Node {

        }

        export class IfStatement extends Node {

        }

        export class ForStatement extends Node {

        }

        export class BlockStatement extends Node {

        }

        export class ProgramExpression extends Node {
          body : Node[];
          constructor(line: number, column : number, body : Node[]) {
            super("ProgramExpression", line, column);
            this.body = body;
          }
        }

        export class ReturnStatement extends Node {
          argument : Node;
          constructor(line: number, column : number, argument? : Node) {
            super("ReturnStatement", line, column);
            this.argument = argument;
          }
          toEsprima() : esprima.Syntax.ReturnStatement {
            return builter.returnStatement()
            return {
              type: "ReturnStatement",
              loc: {
                start: {
                  line: this.line,
                  column: this.column
                },
                end: {
                  line: this.line,
                  column: this.column
                }
              },
              argument: this.argument
            }
          }
        }

        export class ExpressionStatement extends Node {
          expression : Node;
          constructor(line: number, column : number, expression : Node) {
            super("ExpressionStatement", line, column);
            this.expression = expression;
          }
        }
    }
  }
}
