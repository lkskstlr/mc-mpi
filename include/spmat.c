#include "spmat.h"
#include <stdio.h>

void add_node(Controller *controller, float_t value,
              int n_neigh, size_t *neigh) {

    // compute space
    size_t size =
        sizeof(Node) + n_neigh * sizeof(node_ptr_t);

    if (controller->len + size > controller->len_alloc) {
        // must alloc more
        inc_size(controller);
    }

    // add Node
    Node node;
    node.n = controller->N;
    controller->N++;
    node.value = value;
    node.n_neigh = n_neigh;

    *(node_ptr_t)(controller->start + controller->len) =
        node;

    // table
    if (controller->table_len < controller->N) {
        inc_table_size(controller);
    }
    controller->table[controller->N] =
        (node_ptr_t)(controller->start + controller->len);

    controller->len += sizeof(Node);

    // neighbors
    
}

void inc_size(Controller *controller) {
    controller->start =
        realloc(controller->start,
                MULT_LEN * controller->len_alloc);

    if (!controller->start) {
        fprintf(stderr, "Realloc failed.\n");
        exit(EXIT_FAILURE);
    }

    controller->len_alloc *= MULT_LEN;
}

void inc_table_size(Controller *controller) {
    controller->table = realloc(
        controller->table, MULT_LEN * sizeof(node_ptr_t) *
                               controller->table_len);

    if (!controller->table) {
        fprintf(stderr, "Table Realloc failed.\n");
        exit(EXIT_FAILURE);
    }

    controller->table_len *= MULT_LEN;
}