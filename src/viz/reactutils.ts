/// <reference path="./../../typings/tsd.d.ts" />
import React = require("react");

class NotImplementedError implements Error {
    public name = "NotImplementedError";
    public message: string;

    constructor(methodName: string) {
        this.message = methodName + " should be implemented by React";
    }
}
export interface ClassCreator<P, S> {
    (specification: React.Specification<P, S>): React.ReactComponentFactory<P>;
}

export class Component<P, S> implements React.Specification<P, S>, React.Component<P, S> {
    public refs: {
        [key: string]: React.DomReferencer
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

    render(): React.ReactElement<any, any> {
        return null;
    }
}

var ILLEGAL_KEYS: {[key: string]: boolean} = {
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
export interface ComponentClass<P, S> {
    new (): Component<P, S>
}
export function createClass<P, S>(
    createClass: ClassCreator<P, S>,
    clazz: ComponentClass<P, S>): React.ReactComponentFactory<P> {
    var key: string;
    var spec: React.Specification<P, S> = (<React.Specification<P, S>>{});
    spec.displayName = clazz.prototype.constructor.name;
    for (key in clazz.prototype) {
        if (!ILLEGAL_KEYS[key]) {
            (<any>spec)[key] = (<any>clazz.prototype)[key];
        }
    }
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
    return createClass(spec);
}

