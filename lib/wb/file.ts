module lib.wb {
    export function wbImport(state, stack, file:string, numRowsRef) {
        numRowsRef.stack[numRowsRef.id] = 100;
        return _.range(100);
    }
}