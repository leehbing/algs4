package com.leetcode;

import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 16/6/2020 3:07 PM
 * @Author: lihongbing
 */
//297. 二叉树的序列化与反序列化
//序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。
//
//请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列/反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。
//
//示例: 
//你可以将以下二叉树：
//      1
//     / \
//    2   5
//   / \
//  3   4
//
//序列化为 "[1,2,5,3,4,null,null]"
//提示: 这与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。
// 你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。
//
//说明: 不要使用类的成员/全局/静态变量来存储状态，你的序列化和反序列化算法应该是无状态的。

//方法一：深度优先搜索    选择先序遍历的编码方式
//我们从根节点 1 开始，序列化字符串是 1,。然后我们跳到根节点 2 的左子树，序列化字符串变成 1,2,。
// 现在从节点 2 开始，我们访问它的左节点 3（1,2,3,None,None,）和右节点 4(1,2,3,None,None,4,None,None)。
// None,None, 是用来标记缺少左、右子节点，这就是我们在序列化期间保存树结构的方式。
// 最后，我们回到根节点 1 并访问它的右子树，它恰好是叶节点 5。最后，序列化字符串是 1,2,3,None,None,4,None,None,5,None,None,。
//
//即我们可以先序遍历这颗二叉树，遇到空子树的时候序列化成 None，否则继续递归序列化。那么我们如何反序列化呢？首先我们需要根据逗号 , 把原先的序列分割开来得到先序遍历的元素列表，然后从左向右遍历这个序列：
//  如果当前的元素为 None，则当前为空树
//  否则先解析这棵树的左子树，再解析它的右子树
//
//时间复杂度：在序列化和反序列化函数中，我们只访问每个节点一次，因此时间复杂度为 O(n)，其中 n 是节点数，即树的大小。
//空间复杂度：在序列化和反序列化函数中，我们递归会使用栈空间，故渐进空间复杂度为 O(n)。
public class Codec {
    public String rserialize(TreeNode root, String str) {
        if (root == null) {
            str += "None,";
        } else {
            str += str.valueOf(root.val) + ",";
            str = rserialize(root.left, str);
            str = rserialize(root.right, str);
        }
        return str;
    }

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return rserialize(root, "");
    }

    public TreeNode rdeserialize(List<String> l) {
        if (l.get(0).equals("None")) {
            l.remove(0);
            return null;
        }

        TreeNode root = new TreeNode(Integer.valueOf(l.get(0)));
        l.remove(0);
        root.left = rdeserialize(l);
        root.right = rdeserialize(l);

        return root;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] data_array = data.split(",");
        List<String> data_list = new LinkedList<String>(Arrays.asList(data_array));
        return rdeserialize(data_list);
    }
}

//449. 序列化和反序列化二叉搜索树
//序列化是将数据结构或对象转换为一系列位的过程，以便它可以存储在文件或内存缓冲区中，或通过网络连接链路传输，
//以便稍后在同一个或另一个计算机环境中重建。
//
//设计一个算法来序列化和反序列化二叉搜索树。 对序列化/反序列化算法的工作方式没有限制。
// 您只需确保二叉搜索树可以序列化为字符串，并且可以将该字符串反序列化为最初的二叉搜索树。
//
//编码的字符串应尽可能紧凑。
//
//注意：不要使用类成员/全局/静态变量来存储状态。 你的序列化和反序列化算法应该是无状态的。
//
//算法：
//序列化可以很容易的实现，但是对于反序列化我们选择后续遍历会更好。
//感觉给的方法不如上面的297题目写的那样
class Codec2 {

    public StringBuilder postorder(TreeNode root, StringBuilder sb) {
        if (root == null) return sb;
        postorder(root.left, sb);
        postorder(root.right, sb);
        sb.append(root.val);
        sb.append(' ');
        return sb;
    }

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = postorder(root, new StringBuilder());
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    //ArrayDeque是JDK容器中的一个双端队列实现，内部使用数组进行元素存储，不允许存储null值，
    // 可以高效的进行元素查找和尾部插入取出，是用作队列、双端队列、栈的绝佳选择，性能比LinkedList还要好。
    public TreeNode helper(Integer lower, Integer upper, ArrayDeque<Integer> nums) {
        if (nums.isEmpty()) return null;
        int val = nums.getLast();
        if (val < lower || val > upper) return null;

        nums.removeLast();
        TreeNode root = new TreeNode(val);
        root.right = helper(val, upper, nums);
        root.left = helper(lower, val, nums);
        return root;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) return null;
        ArrayDeque<Integer> nums = new ArrayDeque<Integer>();
        for (String s : data.split("\\s+"))
            nums.add(Integer.valueOf(s));
        return helper(Integer.MIN_VALUE, Integer.MAX_VALUE, nums);
    }
}


class Solution {
    //652. 寻找重复的子树
    //给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其中任意一棵的根结点即可。
    //两棵树重复是指它们具有相同的结构以及相同的结点值。
    //
    //示例 1：
    //        1
    //       / \
    //      2   3
    //     /   / \
    //    4   2   4
    //       /
    //      4
    //下面是两个重复的子树：
    //
    //      2
    //     /
    //    4
    //和
    //
    //    4
    //因此，你需要以列表的形式返回上述重复子树的根结点。

    List<TreeNode> ans;
    Map<String, Integer> count;

    //方法一：使用深度优先搜索，其中递归函数返回当前子树的序列化结果。
    // 把每个节点开始的子树序列化结果保存在 map 中，然后判断是否存在重复的子树。
    //时间复杂度：O(N^2)，其中 N 是二叉树上节点的数量。遍历所有节点，在每个节点处序列化需要时间 O(N)
    //空间复杂度：O(N^2)，count 的大小。
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        count = new HashMap();
        ans = new ArrayList();
        collect(root);
        return ans;
    }

    public String collect(TreeNode node) {
        if (node == null) return "#";
        //先序遍历
        String serial = node.val + "," + collect(node.left) + "," + collect(node.right);
        count.put(serial, count.getOrDefault(serial, 0) + 1);
        if (count.get(serial) == 2)
            ans.add(node);
        return serial;
    }

    //方法二：
}

