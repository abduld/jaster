

/*
 *The MIT License (MIT)

 Copyright (c) 2014 Asana

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */


/// <reference path="react.d.ts" />
module lib {
    export module viz {




    // from https://github.com/Asana/typed-react
    export class NotImplementedError implements Error {
        public name = "NotImplementedError";
        public message: string;

        constructor(methodName: string) {
            this.message = methodName + " should be implemented by react";
        }
    }

    export class Mixin<P, S> implements React.Mixin<P, S> {
        public refs: {
            [key: string]: React.Component<any>
        };
        public props: P;
        public state: S;

        getDOMNode(): Element {
            throw new NotImplementedError("getDomNode");
        }

        setState(nextState: S, callback?: () => void): void {
            throw new NotImplementedError("setState");
        }

        replaceState(nextState: S, callback?: () => void): void {
            throw new NotImplementedError("replaceState");
        }

        forceUpdate(callback?: () => void): void {
            throw new NotImplementedError("forceUpdate");
        }

        isMounted(): boolean {
            throw new NotImplementedError("isMounted");
        }

        setProps(nextProps: P, callback?: () => void): void {
            throw new NotImplementedError("setProps");
        }

        replaceProps(nextProps: P, callback?: () => void): void {
            throw new NotImplementedError("replaceProps");
        }
    }

    export class Component<P, S> extends Mixin<P, S> implements React.CompositeComponent<P, S> {
        render(): React.ReactElement<any> {
            return null;
        }
    }

    var ILLEGAL_KEYS: { [key: string]: boolean } = {
        constructor: true,
        refs: true,
        props: true,
        state: true,
        getDOMNode: true,
        setState: true,
        replaceState: true,
        forceUpdate: true,
        isMounted: true,
        setProps: true,
        replaceProps: true
    };

    function extractPrototype<T>(clazz: { new (): T }): T {
        var proto: T = (<T>{});
        for (var key in clazz.prototype) {
            if (ILLEGAL_KEYS[key] === undefined) {
                (<any>proto)[key] = clazz.prototype[key];
            }
        }
        return proto;
    }

    export function createMixin<P, S>(clazz: { new (): Mixin<P, S> }): React.Mixin<P, S> {
        return extractPrototype(clazz);
    }

    export function createClass<P, S>(clazz: { new (): Component<P, S> }, mixins?: React.Mixin<P, S>[]): React.ComponentClass<P> {
        var spec: React.ComponentSpec<P, S> = extractPrototype(clazz);
        spec.displayName = clazz.prototype.constructor.name;
        if (spec.componentWillMount !== undefined) {
            var componentWillMount = spec.componentWillMount;
            spec.componentWillMount = function() {
                clazz.apply(this);
                componentWillMount.apply(this);
            };
        } else {
            spec.componentWillMount = function() {
                clazz.apply(this);
            };
        }
        if (mixins !== undefined && mixins !== null) {
            spec.mixins = mixins;
        }
        return React.createClass(spec);
    }

    }
}
