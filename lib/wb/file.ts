module lib.wb {
    export function wbImport(state, stack, file: string, numRowsRef) {
        var size = 1000;
        if (_.isObject(numRowsRef)) {
            numRowsRef.stack[numRowsRef.id] = size;
        }
        return _.range(size);
    }
}
