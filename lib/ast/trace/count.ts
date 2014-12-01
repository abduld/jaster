

module lib.ast {
    export module trace {
        export module counter {
            var counter_: Map<string, number>;
            if (lib.utils.isUndefined(counter_)) {
                counter_ = new Map<string, number>();;
            }

            export function increment(x: string): number;
            export function increment(x: any): number {
                if (!lib.utils.isString(x)) {
                    return increment(lib.utils.hash(x));
                }
                lib.utils.assert.ok(lib.utils.isString(x), "Input to increment is not a string");
                if (counter_.has(x)) {
                    counter_.set(x, counter_.get(x) + 1);
                } else {
                    counter_.set(x, 1);
                }
                return counter_.get(x);
            }
            export function decrement(x: string): number;
            export function decrement(x: string): number {

                if (!lib.utils.isString(x)) {
                    return increment(lib.utils.hash(x));
                }
                lib.utils.assert.ok(lib.utils.isString(x), "Input to increment is not a string");
                if (counter_.has(x)) {
                    counter_.set(x, counter_.get(x) - 1);
                } else {
                    counter_.set(x, -1);
                }
                return counter_.get(x);
            }
        }
    }
}
