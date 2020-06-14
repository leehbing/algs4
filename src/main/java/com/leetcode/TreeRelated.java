package com.leetcode;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 3/6/2020 5:09 PM
 * @Author: lihongbing
 */

import java.util.*;

/**
 * 树相关的题目
 */
public class TreeRelated {
    //671. 二叉树中第二小的节点
    //给定一个非空特殊的二叉树，每个节点都是正数，并且每个节点的子节点数量只能为 2 或 0。
    // 如果一个节点有两个子节点的话，那么这个节点的值不大于它的子节点的值。 
    //
    //给出这样的一个二叉树，你需要输出所有节点中的第二小的值。如果第二小的值不存在的话，输出 -1 。
    //
    //示例 1:
    //
    //输入:
    //    2
    //   / \
    //  2   5
    //     / \
    //    5   7
    //
    //输出: 5
    //说明: 最小的值是 2 ，第二小的值是 5 。
    //方法一：遍历所有结点，然后排序，取第二个
    //方法二：  算法：
    //让 min1 = root.val。当遍历结点 node，如果 node.val > min1，我们知道在 node 处的子树中的所有值都至少是 node.val，因此在该子树中不此存在第二个最小值。因此，我们不需要搜索这个子树。
    //此外，由于我们只关心第二个最小值 ans，因此我们不需要记录任何大于当前第二个最小值的值，因此与方法 1 不同，我们可以完全不用集合存储数据。
    //复杂度分析
    //
    //时间复杂度：O(N)。其中 N 是给定树中的节点总数。我们最多访问每个节点一次。
    //空间复杂度：O(N)，存储在 ans 和 min1 中的信息为 O(1)，但我们的深度优先搜索可能会在调用堆栈中存储多达 O(h)=O(N) 的信息，其中 h 是树的高度。
    public int findSecondMinimumValue(TreeNode root) {
        if (root == null) return -1;
        return help(root, root.val); //root.val是最小的结点
    }

    private int help(TreeNode root, int min) {
        if (root == null) return -1;
        if (root.val > min) return root.val;
        int left = help(root.left, min);
        int right = help(root.right, min);
        if (left == -1) return right;
        if (right == -1) return left;
        return Math.min(left, right);
    }

    //872. 叶子相似的树
    //请考虑一颗二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列
    //    3
    //   /  \
    //  5    1
    // / \   / \
    // 6  2  9  8
    //   / \
    //  7   4
    //举个例子，如上图所示，给定一颗叶值序列为 (6, 7, 4, 9, 8) 的树。
    //
    //如果有两颗二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。
    //
    //如果给定的两个头结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {

    }

    //104. 二叉树的最大深度===>就是高度
    //给定一个二叉树，找出其最大深度。
    //时间复杂度：我们每个结点只访问一次，因此时间复杂度为 O(N),其中 N 是结点的数量。
    //空间复杂度：在最糟糕的情况下，树是完全不平衡的，例如每个结点只剩下左子结点，递归将会被调用 N 次（树的高度），因此保持调用栈的存储将是 O(N)。但在最好的情况下（树是完全平衡的），树的高度将是log(N)。因此，在这种情况下的空间复杂度将是 O(log(N))。
    //
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }

    //111. 二叉树的最小深度
    //给定一个二叉树，找出其最小深度。
    //最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    //说明: 叶子节点是指没有子节点的节点。
    //时间复杂度：我们访问每个节点一次，时间复杂度为 O(N) ，其中 N 是节点个数。
    //空间复杂度：最坏情况下，整棵树是非平衡的，例如每个节点都只有一个孩子，递归会调用 N （树的高度）次，因此栈的空间开销是 O(N) 。但在最好情况下，树是完全平衡的，高度只有 log(N)，因此在这种情况下空间复杂度只有 O(log(N)) 。
    //
    //
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = minDepth(root.left);
        int rightDepth = minDepth(root.right);
        if (root.left == null) {
            return 1 + rightDepth;
        } else if (root.right == null) {
            return 1 + leftDepth;
        } else {
            return Math.min(leftDepth, rightDepth) + 1;
        }
    }

    //559. N叉树的最大深度
    //给定一个 N 叉树，找到其最大深度。
    //最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
    //
    //时间复杂度：每个节点遍历一次，所以时间复杂度是 O(N)，其中 N 为节点数。
    //空间复杂度：最坏情况下, 树完全非平衡，例如 每个节点有且仅有一个孩子节点，递归调用会发生 N 次（等于树的深度），所以存储调用栈需要 O(N)。
    //但是在最好情况下（树完全平衡），树的高度为 log(N)。
    //所以在此情况下空间复杂度为O(log(N))。
    //
    public int maxDepth(Node root) {
        if (root == null) {
            return 0;
        } else if (root.children.isEmpty()) {
            return 1;
        } else {
            List<Integer> heights = new LinkedList<>();
            for (Node item : root.children) {
                heights.add(maxDepth(item));
            }
            return Collections.max(heights) + 1;
        }
    }

    class Node {
        public int val;
        public List<Node> children;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    }

    ;


    //110. 平衡二叉树
    //给定一个二叉树，判断它是否是高度平衡的二叉树。
    //
    //本题中，一棵高度平衡二叉树定义为：
    //一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

    //方法一：自顶向下的递归
    //算法
    //
    //定义方法 height，用于计算任意一个节点 p ∈ T 的高度：
    //接下来就是比较每个节点左右子树的高度。在一棵以 r 为根节点的树T 中，只有每个节点左右子树高度差不大于 1 时，该树才是平衡的。因此可以比较每个节点左右两棵子树的高度差，然后向上递归。
    //时间复杂度：O(nlogn)
    //空间复杂度：O(n)
    public boolean isBalanced(TreeNode root) {
        // An empty tree satisfies the definition of a balanced tree
        if (root == null) {
            return true;
        }

        // Check if subtrees have height within 1. If they do, check if the
        // subtrees are balanced
        return Math.abs(height(root.left) - height(root.right)) < 2
                && isBalanced(root.left)
                && isBalanced(root.right);
    }

    // Recursively obtain the height of a tree. An empty tree has -1 height
    private int height(TreeNode root) {
        // An empty tree has height -1
        if (root == null) {
            return -1;
//            return 0; //空树的高度应该是0，反正这道题返回0或者-1都是一样的
        }
        return 1 + Math.max(height(root.left), height(root.right));
    }

    //方法二：自底向上的递归
    //思路
    //方法一计算 height 存在大量冗余。每次调用height 时，要同时计算其子树高度。但是自底向上计算，每个子树的高度只会计算一次。
    // 可以递归先计算当前节点的子节点高度，然后再通过子节点高度判断当前节点是否平衡，从而消除冗余。
    //
    //算法
    //使用与方法一中定义的height 方法。自底向上与自顶向下的逻辑相反，首先判断子树是否平衡，然后比较子树高度判断父节点是否平衡。算法如下：
    //检查子树是否平衡。如果平衡，则使用它们的高度判断父节点是否平衡，并计算父节点的高度。
    //时间复杂度：O(n)，计算每棵子树的高度和判断平衡操作都在恒定时间内完成
    //空间复杂度：O(n)，如果树不平衡，递归栈可能达到 O(n)
    //
    // Utility class to store information from recursive calls
    final class TreeInfo {
        public final int height;
        public final boolean balanced;

        public TreeInfo(int height, boolean balanced) {
            this.height = height;
            this.balanced = balanced;
        }
    }

    // Return whether or not the tree at root is balanced while also storing
    // the tree's height in a reference variable.
    private TreeInfo isBalancedTreeHelper(TreeNode root) {
        // An empty tree is balanced and has height = -1
        if (root == null) {
            return new TreeInfo(-1, true);
        }

        // Check subtrees to see if they are balanced.
        TreeInfo left = isBalancedTreeHelper(root.left);
        if (!left.balanced) {
            return new TreeInfo(-1, false); //不平衡直接返回了
        }
        TreeInfo right = isBalancedTreeHelper(root.right);
        if (!right.balanced) {
            return new TreeInfo(-1, false);
        }

        // Use the height obtained from the recursive calls to
        // determine if the current node is also balanced.
        if (Math.abs(left.height - right.height) < 2) {
            return new TreeInfo(Math.max(left.height, right.height) + 1, true);
        }
        return new TreeInfo(-1, false);
    }

    public boolean isBalanced2(TreeNode root) {
        return isBalancedTreeHelper(root).balanced;
    }

    //101. 对称二叉树
    //给定一个二叉树，检查它是否是镜像对称的。
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    public boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }


    //98. 验证二叉搜索树
    //定义：如果该二叉树的左子树不为空，则左子树上所有节点的值均小于它的根节点的值； 若它的右子树不空，则右子树上所有节点的值均大于它的根节点的值；它的左右子树也为二叉搜索树。
    //方法一：中序遍历有序即可
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    public static boolean isValidBST(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderScan(root, res);
        if (res.size() <= 1) return true;
        for (int i = 1; i < res.size(); i++) {
            if (res.get(i - 1) >= res.get(i)) return false;
        }
        return true;

    }


    //94. 二叉树的中序遍历
    //递归很简单，但是题目要求用迭代来解决
    //方法一：递归
    //时间复杂度：O(n)。递归函数 T(n) = 2*T(n/2)+1
    //空间复杂度：递归的底层用栈来存储之后需要再次访问的节点，最坏情况下需要空间O(n)，平均情况为O(logn)。
    public static List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderScan(root, res);
        return res;
    }

    public static void inorderScan(TreeNode node, List<Integer> res) {
        if (node == null) return;
        inorderScan(node.left, res);
        res.add(node.val);
        inorderScan(node.right, res);
    }

    //方法二：基于栈的遍历，迭代,本质上是在模拟递归
    //本方法的策略就是模拟递归的过程的，使用了栈。
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) { //尽可能的将这个节点的左子树压入Stack，此时栈顶的元素是最左侧的元素，其目的是找到一个最小单位的子树(也就是最左侧的一个节点)，并且在寻找的过程中记录了来源，才能返回上层,同时在返回上层的时候已经处理完毕左子树了。。
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            res.add(curr.val); //当处理完最小单位的子树时，返回到上层处理了中间节点。（如果把整个左中右的遍历都理解成子树的话，就是处理完 左子树->中间(就是一个节点)->右子树）
            curr = curr.right;//如果有右节点，其也要进行中序遍历
        }
        return res;
    }
    //方法三：二叉树的莫里斯遍历  ，将空间复杂度降到O(1)      ==》没看


    //230. 二叉搜索树中第K小的元素
    //给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
    //说明：
    //你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。
    //
    //示例 1:
    //输入: root = [3,1,4,null,2], k = 1
    //   3
    //  / \
    // 1   4
    //  \
    //   2
    //输出: 1
    //进阶：
    //如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化 kthSmallest 函数？
    //方法一：中序遍历，然后求第k的值
    //时间复杂度：O(N)，遍历了整个树。
    //空间复杂度：O(N)，用了一个数组存储中序序列。
    public int kthSmallest(TreeNode root, int k) {
        List<Integer> res = new ArrayList<>();
        inorderScan(root, res);
        return res.get(k - 1);
    }

    //方法二：迭代
    //算法：
    //在栈的帮助下，可以将方法一的递归转换为迭代，这样可以加快速度，因为这样可以不用遍历整个树，可以在找到答案后停止。
    //时间复杂度：O(H+k)，其中 H 指的是树的高度，由于我们开始遍历之前，要先向下达到叶，当树是一个平衡树时：复杂度为O(logN+k)。当树是一个不平衡树时：复杂度为 O(N+k)，此时所有的节点都在左子树。
    //空间复杂度：O(H+k)。当树是一个平衡树时：O(logN+k)。当树是一个非平衡树时：O(N+k)。
    //
    public int kthSmallest2(TreeNode root, int k) {
        LinkedList<TreeNode> stack = new LinkedList<TreeNode>();
        while (true) {
            while (root != null) {
                stack.add(root);
                root = root.left;
            }
            root = stack.removeLast();
            if (--k == 0) return root.val;
            root = root.right;
        }
    }


    //144. 二叉树的前序遍历
    //方法一：递归
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        preorderScan(root, res);
        return res;
    }

    public static void preorderScan(TreeNode node, List<Integer> res) {
        if (node == null) return;
        res.add(node.val);
        preorderScan(node.left, res);
        preorderScan(node.right, res);
    }

    //方法二：基于栈的遍历， 迭代, 本质上是在模拟递归
    //时间复杂度：访问每个节点恰好一次，时间复杂度为 O(N) ，其中 N 是节点的个数，也就是树的大小。
    //空间复杂度：取决于树的结构，最坏情况存储整棵树，因此空间复杂度是 O(N)。
    public List<Integer> preorderTraversal2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        LinkedList<Integer> output = new LinkedList<>();
        if (root == null) {
            return output;
        }
        stack.add(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            output.add(node.val);
            if (node.right != null) { //先打印左子树，然后右子树。所以先加入Stack的就是右子树，然后左子树
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return output;
    }

    //方法三：二叉树的莫里斯遍历(前序/中序/后序都可以)，将空间复杂度降到O(1)      ==》没看

    //145. 二叉树的后序遍历
    //方法一：递归
    public static List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        postorderScan(root, res);
        return res;
    }

    public static void postorderScan(TreeNode node, List<Integer> res) {
        if (node == null) return;
        postorderScan(node.left, res);
        postorderScan(node.right, res);
        res.add(node.val);
    }

    //方法二：迭代
    //从根节点开始依次迭代，弹出栈顶元素输出到输出列表中，然后依次压入它的所有孩子节点，按照从上到下、从左至右的顺序依次压入栈中。
    //因为深度优先搜索后序遍历的顺序是从下到上、从左至右，所以需要将输出列表逆序输出。
    public List<Integer> postorderTraversal2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        LinkedList<Integer> output = new LinkedList<>();
        if (root == null) {
            return output;
        }
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            output.addFirst(node.val);
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
        return output;
    }

    //105. 从前序与中序遍历序列构造二叉树
    //根据一棵树的前序遍历与中序遍历构造二叉树。
    //注意:
    //你可以假设树中没有重复的元素。

    //前序遍历：     [根节点], [         左子树的前序遍历结果        ], [         右子树的前序遍历结果          ]
    //                 ↑      ↑                                 ↑    ↑                                  ↑
    //              preleft preleft+1        pIndex-inLeft+preLeft  pIndex-inLeft+preLeft+1         preRight
    //
    //中序遍历：     [         左子树的前序遍历结果        ], [根节点], [         右子树的前序遍历结果          ]
    //               ↑                                ↑      ↑       ↑                                  ↑
    //              inleft                       pIndex-1   pIndex  pIndex+1                         inRight
    //
    //要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目。
    // 由于同一颗子树的前序遍历和中序遍历的长度显然是相同的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。
    //这样一来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，
    // 我们就可以递归地对构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。
    //细节：在中序遍历中对根节点进行定位时，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，
    // 但这样做的时间复杂度较高。我们可以考虑使用哈希映射（HashMap）来帮助我们快速地定位根节点。
    // 对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。
    // 在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 O(1) 的时间对根节点进行定位了。
    //
    //时间复杂度：O(n)，其中 n 是树中的节点个数。
    //空间复杂度：O(n)，除去返回的答案需要的O(n) 空间之外，我们还需要使用 O(n) 的空间存储哈希映射，以及 O(h)（其中 h 是树的高度）的空间表示递归时栈空间。这里 h < n，所以总空间复杂度为 O(n)。
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int preLen = preorder.length;
        int inLen = inorder.length;
        if (preLen != inLen) {
            throw new RuntimeException("Incorrect input data.");
        }
        // 构造哈希映射，帮助我们快速定位根节点
        Map<Integer, Integer> map = new HashMap<Integer, Integer>(preLen);
        for (int i = 0; i < inLen; i++) {
            map.put(inorder[i], i);
        }
        return buildTree(preorder, 0, preLen - 1, map, 0, inLen - 1);
    }

    public TreeNode buildTree(int[] preorder, int preLeft, int preRight, Map<Integer, Integer> map, int inLeft, int inRight) {
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        // 前序遍历中的第一个节点就是根节点
        int rootVal = preorder[preLeft];
        // 在中序遍历中定位根节点
        int pIndex = map.get(rootVal);

        // 先把根节点建立出来
        TreeNode root = new TreeNode(rootVal);

        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = buildTree(preorder, preLeft + 1, pIndex - inLeft + preLeft, map, inLeft, pIndex - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = buildTree(preorder, pIndex - inLeft + preLeft + 1, preRight, map, pIndex + 1, inRight);
        return root;
    }
}
