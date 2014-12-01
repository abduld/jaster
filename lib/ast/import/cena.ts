

module lib.ast {
  export module import {
  export module cena {
        export interface Location {
            row: number;
            column: number;
        }
        export class Node {
            loc: Location;
            constructor(loc: Location) {
                this.loc = loc;
            }
        }
        export class LiteralNode<T> extends Node {
            value: T;
            constructor(loc: Location) {
                super(loc);
            }
        }
        export class BooleanNode extends LiteralNode<boolean> {
            constructor() { }
        }
    }



  }
}
