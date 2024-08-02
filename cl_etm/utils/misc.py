def display_dglgraph_info(graph, subject_id=None):

    if not subject_id:
        for subject_id in graph:
            subgraph = graph[subject_id]
            display_graph(subgraph, subject_id=subject_id)

    else:
        display_graph(graph, subject_id=subject_id)

def display_graph(graph, subject_id=None):
    print(f"\nSubject ID: {subject_id}")
    print("-" * 30)

    # Displaying nodes
    print("Nodes:")
    for ntype in graph.ntypes:
        num_nodes = graph.num_nodes(ntype)
        print(f"  {ntype.capitalize()}: {num_nodes}")

    # Displaying edges
    print("\nEdges:")
    for etype in graph.etypes:
        src_type, _, dst_type = graph.to_canonical_etype(etype)
        num_edges = graph.num_edges(etype)
        print(f"  {src_type.capitalize()} -> {dst_type.capitalize()} ({etype}): {num_edges}")
    
    print("=" * 30)