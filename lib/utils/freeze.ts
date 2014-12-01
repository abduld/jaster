

module lib.utils {
    export function freeze(o: any): any {
        Object.freeze(o);
        Object.getOwnPropertyNames(o).forEach(function(prop) {
            if (o.hasOwnProperty(prop)
                && o[prop] !== null
                && (typeof o[prop] === "object" || typeof o[prop] === "function")
                && !Object.isFrozen(o[prop])) {
                freeze(o[prop]);
            }
        });
        return o;
    }
}