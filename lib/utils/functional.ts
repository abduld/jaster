

module lib.utils {
    export module functional {

        export function forEach<T, U>(array: T[], callback: (element: T) => U): U {
            var result: U;
            if (array) {
                for (var i = 0, len = array.length; i < len; i++) {
                    if (result = callback(array[i])) {
                        break;
                    }
                }
            }
            return result;
        }

        export function contains<T>(array: T[], value: T): boolean {
            if (array) {
                for (var i = 0, len = array.length; i < len; i++) {
                    if (array[i] === value) {
                        return true;
                    }
                }
            }
            return false;
        }

        export function indexOf<T>(array: T[], value: T): number {
            if (array) {
                for (var i = 0, len = array.length; i < len; i++) {
                    if (array[i] === value) {
                        return i;
                    }
                }
            }
            return -1;
        }

        export function countWhere<T>(array: T[], predicate: (x: T) => boolean): number {
            var count = 0;
            if (array) {
                for (var i = 0, len = array.length; i < len; i++) {
                    if (predicate(array[i])) {
                        count++;
                    }
                }
            }
            return count;
        }

        export function filter<T>(array: T[], f: (x: T) => boolean): T[] {
            if (array) {
                var result: T[] = [];
                for (var i = 0, len = array.length; i < len; i++) {
                    var item = array[i];
                    if (f(item)) {
                        result.push(item);
                    }
                }
            }
            return result;
        }

        export function map<T, U>(array: T[], f: (x: T) => U): U[] {
            if (array) {
                var result: U[] = [];
                for (var i = 0, len = array.length; i < len; i++) {
                    result.push(f(array[i]));
                }
            }
            return result;
        }

        export function concatenate<T>(array1: T[], array2: T[]): T[] {
            if (!array2 || !array2.length) return array1;
            if (!array1 || !array1.length) return array2;

            return array1.concat(array2);
        }

        export function deduplicate<T>(array: T[]): T[] {
            if (array) {
                var result: T[] = [];
                for (var i = 0, len = array.length; i < len; i++) {
                    var item = array[i];
                    if (!contains(result, item)) result.push(item);
                }
            }
            return result;
        }

        export function sum(array: any[], prop: string): number {
            var result = 0;
            for (var i = 0; i < array.length; i++) {
                result += array[i][prop];
            }
            return result;
        }

        /**
         * Returns the last element of an array if non-empty, undefined otherwise.
         */
        export function lastOrUndefined<T>(array: T[]): T {
            if (array.length === 0) {
                return undefined;
            }

            return array[array.length - 1];
        }

        export function binarySearch(array: number[], value: number): number {
            var low = 0;
            var high = array.length - 1;

            while (low <= high) {
                var middle = low + ((high - low) >> 1);
                var midValue = array[middle];

                if (midValue === value) {
                    return middle;
                }
                else if (midValue > value) {
                    high = middle - 1;
                }
                else {
                    low = middle + 1;
                }
            }

            return ~low;
        }

        var hasOwnProperty = Object.prototype.hasOwnProperty;

        export function hasProperty<T>(map: Map<string, T>, key: string): boolean {
            return hasOwnProperty.call(map, key);
        }

        export function getProperty<T>(map: Map<string, T>, key: string): T {
            return hasOwnProperty.call(map, key) ? map[key] : undefined;
        }

        export function isEmpty<T>(map: Map<string, T>) {
            for (var id in map) {
                if (hasProperty(map, id)) {
                    return false;
                }
            }
            return true;
        }

        export function clone<T>(object: T): T {
            var result: any = {};
            for (var id in object) {
                result[id] = (<any>object)[id];
            }
            return <T>result;
        }

        export function forEachValue<T, U>(map: Map<string, T>, callback: (value: T) => U): U {
            var result: U;
            for (var id in map) {
                if (result = callback(map[id])) break;
            }
            return result;
        }

        export function forEachKey<T, U>(map: Map<string, T>, callback: (key: string) => U): U {
            var result: U;
            for (var id in map) {
                if (result = callback(id)) break;
            }
            return result;
        }

        export function lookUp<T>(map: Map<string, T>, key: string): T {
            return hasProperty(map, key) ? map[key] : undefined;
        }

        export function mapToArray<T>(map: Map<string, T>): T[] {
            var result: T[] = [];

            for (var id in map) {
                result.push(map[id]);
            }

            return result;
        }
    }
}