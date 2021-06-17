function left_right_score(obj, node_id)
    this_node = obj.nodes{node_id};
    is_terminal = this_node.child_left < 0;
    if is_terminal
        return;
    end

    n_left = this_node.final_score * this_node.depth;
    obj.nodes{this_node.child_left}.final_score = (n_left + 1) / (this_node.depth + 1);
    obj.nodes{this_node.child_right}.final_score = (n_left) / (this_node.depth + 1);

    left_right_score(this_node.child_left);
    left_right_score(this_node.child_right);
end