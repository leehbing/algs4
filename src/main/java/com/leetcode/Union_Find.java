package com.leetcode;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 17/6/2020 3:31 PM
 * @Author: lihongbing
 */
public class Union_Find {
    //990. 等式方程的可满足性
    //给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，
    // 并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
    //
    //只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
    //
    //我们可以将每一个变量看作图中的一个节点，把相等的关系 == 看作是连接两个节点的边，那么由于表示相等关系的等式方程具有传递性，即如果 a==b 和 b==c 成立，则 a==c 也成立。
    // 也就是说，所有相等的变量属于同一个连通分量。因此，我们可以使用并查集(union-find算法)来维护这种连通分量的关系。
    //首先遍历所有的等式，构造并查集。同一个等式中的两个变量属于同一个连通分量，因此将两个变量进行合并。
    //然后遍历所有的不等式。同一个不等式中的两个变量不能属于同一个连通分量，因此对两个变量分别查找其所在的连通分量，如果两个变量在同一个连通分量中，则产生矛盾，返回 false。
    //如果遍历完所有的不等式没有发现矛盾，则返回 true。
    //见《算法》的第一章的union-find算法
    //路径压缩的quick-union算法，当然还有好几种union-find 算法
    //时间复杂度：O(n+ClogC)，其中 n 是 equations 中的方程数量，C 是变量的总数，在本题中变量都是小写字母，即 C≤26。
    // 上面的并查集代码中使用了路径压缩优化，对于每个方程的合并和查找的均摊时间复杂度都是 O(logC)。由于需要遍历每个方程，因此总时间复杂度是 O(n+ClogC)。
    //
    //空间复杂度：O(C)。创建一个数组 parent 存储每个变量的连通分量信息，由于变量都是小写字母，因此 parent 是长度为 C。
    public boolean equationsPossible(String[] equations) {
        int length = equations.length;
        int[] parent = new int[26]; //parent of i，触点i的父结点
        for (int i = 0; i < 26; i++) {
            parent[i] = i;
        }
        for (String str : equations) {
            if (str.charAt(1) == '=') {
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                union(parent, index1, index2);
            }
        }
        for (String str : equations) {
            if (str.charAt(1) == '!') {
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                if (find(parent, index1) == find(parent, index2)) {
                    return false;
                }
            }
        }
        return true;
    }

    public void union(int[] parent, int index1, int index2) {
        parent[find(parent, index1)] = find(parent, index2);
    }

    public int find(int[] parent, int index) { //路径压缩算法
        while (parent[index] != index) {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    }
}
