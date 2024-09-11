from priority_dict import *

def dijkstras_search(origin_key, goal_key, graph):
    
    # The priority queue of open vertices we've reached.
    # Keys are the vertex keys, vals are the distances.
    open_queue = priority_dict.priority_dict({})
    
    # The dictionary of closed vertices we've processed.
    closed_dict = {}
    
    # The dictionary of predecessors for each vertex.
    predecessors = {}
    
    # Add the origin to the open queue.
    open_queue[origin_key] = 0.0

    # Iterate through the open queue, until we find the goal.
    # Each time, perform a Dijkstra's update on the queue.
    # TODO: Implement the Dijstra update loop.
    goal_found = False
    while (open_queue):
        # Find edges that are connected to current edge
        u, u_dist = priority_dict.pop_smallest()
        if u==goal_key:
            goal_found = True
        for edge in graph.out_edges([u], data=True):
            v = edge[1]
            uv_dist = edge[2]['length']
            if v in closed_dict:
                continue
            else:
                v_dist = u_dist + uv_dist
                if v in open_queue:
                    if v_dist<open_queue[v]:
                        open_queue[v] = v_dist
                        predecessors[v] = u
                else:
                    open_queue[v] = v_dist
                    predecessors[v] = u
            closed_dict.add(u)

    # If we get through entire priority queue without finding the goal,
    # something is wrong.
    if not goal_found:
        raise ValueError("Goal not found in search.")
    
    # Construct the path from the predecessors dictionary.
    return get_path(origin_key, goal_key, predecessors)      