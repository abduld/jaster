/// <reference path="../typings/tsd.d.ts" />
/// <reference path="utils.ts" />

import React = require("react");
//import utils = require("./utils");

class NotImplementedError implements Error {
    public name = "NotImplementedError";
    public message: string;

    constructor(methodName: string) {
        this.message = methodName + " should be implemented by React";
    }
}
interface ClassCreator<P, S> {
    (specification: React.Specification<P, S>): React.ReactComponentFactory<P>;
}

class Component<P, S> implements React.Specification<P, S>, React.Component<P, S> {
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
interface ComponentClass<P, S> {
    new (): Component<P, S>
}
function createClass<P, S>(
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

export interface CellProps {
    threadGroup : string;
}

interface CellState {
    activated : boolean;
}

class Cell extends Component<CellProps, CellState> {
    //private draw = SVG(utils.guuid()).size(100, 100);

    getInitialState() {
        return {
            activated: false
        };
    }

    activate() {
        this.setState({
            activated: true
        });
    }

    private getFill() : string {
        if (this.state.activated) {
            return "#fff";
        } else {
            return "#000";
        }
    }
    render() {
        return React.DOM.rect({width: 100, height: 100, fill: this.getFill()});
    }
}
export var cell = createClass<CellProps, CellState> (React.createClass, Cell);

//import chai = require("chai");

//var expect = chai.expect;

//expect(React.renderToStaticMarkup(cell({threadGroup: utils.guuid()}))).to.equal(false);

