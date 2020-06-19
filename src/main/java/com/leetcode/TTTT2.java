package com.leetcode;


import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 20/5/2020 3:43 PM
 * @Author: lihongbing
 */
public class TTTT2 {


    //198. 打家劫舍
    //给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
    //典型的动态规划
    public static int rob(int[] nums) {
        //方法，因为都是非负整数，所以，最高金额要么是所有索引为偶数的数据之和，要么是所有索引为奇数的数据===>不对，万一跳两个后，[2,1,1,2]=>2+2=4
//        int oushu = 0;
//        int jishu = 0;
//        for (int i = 0; i < nums.length; i++) {
//            if (i % 2 == 0) oushu += nums[i];
//            else jishu += nums[i];
//        }
//        return oushu > jishu ? oushu : jishu;

        //官方解答： 动态规划 + 滚动数组

        //首先考虑最简单的情况。如果只有一间房屋，则偷窃该房屋，可以偷窃到最高总金额。
        // 如果只有两间房屋，则由于两间房屋相邻，不能同时偷窃，只能偷窃其中的一间房屋，因此选择其中金额较高的房屋进行偷窃，可以偷窃到最高总金额。
        //如果房屋数量大于两间，应该如何计算能够偷窃到的最高总金额呢？对于第 k (k>2) 间房屋，有两个选项：
        //  1.偷窃第 k 间房屋，那么就不能偷窃第 k−1 间房屋，偷窃总金额为前 k−2 间房屋的最高总金额与第 k 间房屋的金额之和。
        //  2.不偷窃第 k 间房屋，偷窃总金额为前 k−1 间房屋的最高总金额。
        //
        //在两个选项中选择偷窃总金额较大的选项，该选项对应的偷窃总金额即为前 k 间房屋能偷窃到的最高总金额。
        //
        //用 dp[i] 表示前 i 间房屋能偷窃到的最高总金额，那么就有如下的状态转移方程：
        //dp[i]=max(dp[i−2]+nums[i],dp[i−1])
        //
        //边界条件为：
        //dp[0]=nums[0]         只有一间房屋，则偷窃该房屋
        //dp[1]=max(nums[0],nums[1])    只有两间房屋，选择其中金额较高的房屋进行偷窃
        //
        //最终的答案即为 dp[n−1]，其中 n 是数组的长度
        //时间复杂度：O(n)
        //空间复杂度：O(n)
        //自己的疑问：如果金额有负数怎么办呢？？
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int length = nums.length;
        if (length == 1) {
            return nums[0];
        }
        int[] dp = new int[length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[length - 1];
        //上述方法使用了数组存储结果。考虑到每间房屋的最高总金额只和该房屋的前两间房屋的最高总金额相关，因此可以使用滚动数组，
        // 在每个时刻只需要存储前两间房屋的最高总金额,这样，空间复杂度只为O(1)
        //        if (nums == null || nums.length == 0) {
        //            return 0;
        //        }
        //        int length = nums.length;
        //        if (length == 1) {
        //            return nums[0];
        //        }
        //        int first = nums[0], second = Math.max(nums[0], nums[1]);
        //        for (int i = 2; i < length; i++) {
        //            int temp = second;
        //            second = Math.max(first + nums[i], second);
        //            first = temp;
        //        }
        //        return second;
    }

    //213. 打家劫舍 II  所有的房屋都围成一圈
    //环状排列意味着第一个房子和最后一个房子中只能选择一个偷窃，因此可以把此环状排列房间问题约化为两个单排排列房间子问题：
    //
    //1.在不偷窃第一个房子的情况下（即 nums[1:]]），最大金额是 p1
    //2.在不偷窃最后一个房子的情况下（即 nums[:n−1]），最大金额是 p2
    // 综合偷窃最大金额： 为以上两种情况的较大值，即 max(p1,p2)max(p1,p2) 。
    //问题转化不太严谨的，因为对于一个环来说，如果求最大值，存在首尾两个节点都不取的情况；
    //但为什么问题可以转化为求两个队列呢？
    //因为对于上述情况，即首尾都不取时，它的最大值肯定小于等于只去掉首或者只去掉尾的队列,即f（n1,n2,n3）<=f(n1,n2,n3,n4)
    public static int rob2(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)),
                myRob(Arrays.copyOfRange(nums, 1, nums.length)));
    }

    private static int myRob(int[] nums) {
        int pre = 0, cur = 0, tmp;
        for (int num : nums) {
            tmp = cur;
            cur = Math.max(pre + num, cur);
            pre = tmp;
        }
        return cur;
    }

    //337. 打家劫舍 III
    //房子排列成二叉树的形状：
    //输入: [3,2,3,null,3,null,1]
    //
    //     3
    //    / \
    //   2   3
    //    \   \
    //     3   1
    //
    //输出: 7
    //解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
    //本题目本身就是动态规划的树形版本，通过此题解，可以了解一下树形问题在动态规划问题解法
    //我们通过三个方法不断递进解决问题
    //
    //解法一通过递归实现，虽然解决了问题，但是复杂度太高
    //解法二通过解决方法一中的重复子问题，实现了性能的百倍提升
    //解法三直接省去了重复子问题，性能又提升了一步
    //
    //
    public static int rob(TreeNode root) {
        return 0;
    }


    //面试题46. 把数字翻译成字符串
    //给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。
    // 一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
    //
    //
    //方法一：动态规划
    //思路和算法
    //首先我们来通过一个例子理解一下这里「翻译」的过程：我们来尝试翻译「14021402」。
    //
    //分成两种情况：
    //
    //首先我们可以把每一位单独翻译，即 [1,4,0,2]，翻译的结果是 beac
    //然后我们考虑组合某些连续的两位：
    //[14,0,2]，翻译的结果是 oac。
    //[1,40,2]，这种情况是不合法的，因为 40 不能翻译成任何字母。
    //[1,4,02]，这种情况也是不合法的，含有前导零的两位数不在题目规定的翻译规则中，那么 [14,02] 显然也是不合法的。
    //那么我们可以归纳出翻译的规则，字符串的第 i 位置：
    //可以单独作为一位来翻译
    //如果第i−1 位和第 i 位组成的数字在 10 到 25 之间，可以把这两位连起来翻译
    //到这里，我们发现它和「198. 打家劫舍」非常相似。我们可以用 f(i) 表示以第 i 位结尾的前缀串翻译的方案数，考虑第 i 位单独翻译和与前一位连接起来再翻译对 f(i) 的贡献。单独翻译对 f(i) 的贡献为 f(i−1)；如果第 i−1 位存在，并且第 i−1 位和第 i 位形成的数字 x 满足 10≤x≤25，那么就可以把第 i−1 位和第 i 位连起来一起翻译，对 f(i) 的贡献为 f(i−2)，否则为 0。我们可以列出这样的动态规划转移方程：
    //f(i)=f(i−1)+f(i−2)[i−1≥0,10≤x≤25]
    //边界条件是 f(−1)=0，f(0)=1。方程中 [c] 的意思是 c 为真的时候[c]=1，否则[c]=0。
    //有了这个方程我们不难给出一个时间复杂度为 O(n)，空间复杂度为 O(n) 的实现。考虑优化空间复杂度：这里的 f(i) 只和它的前两项f(i−1) 和f(i−2) 相关，
    // 我们可以运用「滚动数组」思想把 f 数组压缩成三个变量，这样空间复杂度就变成了O(1)。
    //时间复杂度：循环的次数是 num 的位数，故渐进时间复杂度为 O(logn)。
    //空间复杂度：虽然这里用了滚动数组，动态规划部分的空间代价是O(1) 的，但是这里用了一个临时变量把数字转化成了字符串，故渐进空间复杂度也是O(logn)。
    //
    public int translateNum(int num) {
        String src = String.valueOf(num);
        int p = 0, q = 0, r = 1;
        for (int i = 0; i < src.length(); ++i) {
            p = q;
            q = r;
            r = 0;
            r += q;
            if (i == 0) {
                continue;
            }
            String pre = src.substring(i - 1, i + 1);
            if (pre.compareTo("25") <= 0 && pre.compareTo("10") >= 0) {
                r += p;
            }
        }
        return r;
    }

    //1431. 拥有最多糖果的孩子
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = candies[0];
        for (int i = 1; i < candies.length; i++) {
            if (max < candies[i]) max = candies[i];
        }
        List<Boolean> result = new ArrayList<>();
        for (int candy : candies) {
            if (candy + extraCandies >= max) {
                result.add(true);
            } else {
                result.add(false);
            }
        }

        return result;
    }


    //70. 爬楼梯
    //假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
    //每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
    //注意：给定 n 是一个正整数。
    //
    //示例 1：
    //输入： 2
    //输出： 2
    //解释： 有两种方法可以爬到楼顶。
    //1.  1 阶 + 1 阶
    //2.  2 阶
    //示例 2：
    //
    //输入： 3
    //输出： 3
    //解释： 有三种方法可以爬到楼顶。
    //1.  1 阶 + 1 阶 + 1 阶
    //2.  1 阶 + 2 阶
    //3.  2 阶 + 1 阶
    //
    //很明显用动态规划方法
    //f(x)=f(x-1)+f(x-2)
    //踩第x台阶的方法 = 第x踩 + 第x不踩(则x-1必须踩)
    //          f(x)=   f(x-1) +  f(x-2)
    //f(0)=1,f(1)=1,f(2)=2,f(3)=3,f(4)=5
    //时间复杂度：循环执行 n 次，每次花费常数的时间代价，故渐进时间复杂度为 O(n)
    //空间复杂度：这里只用了常数个变量作为辅助空间，故渐进空间复杂度为 O(1),「滚动数组思想」把空间复杂度优化成 O(1)
    public int climbStairs(int n) {
        int p = 0, q = 0, r = 1;
        for (int i = 1; i <= n; ++i) {
            p = q;
            q = r;
            r = p + q;
        }
        return r;
    }

    //746. 使用最小花费爬楼梯
    //数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
    //
    //每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
    //
    //您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。
    //
    //示例 1:
    //输入: cost = [10, 15, 20]
    //输出: 15
    //解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
    // 示例 2:
    //输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
    //输出: 6
    //解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。
    //动态规划
    //f(i)=min(f(i-1)+cost[i],f(i-2)+cost[i-1])
    //            踩第n阶          不踩第n阶
    //f(0)=0,f(1)=min(cost[0],0)=0,f(2)=min(cost[0],cost[1])
    public int minCostClimbingStairs(int[] cost) {
        int p = 0, q = 0, r = 0;
        for (int i = 1; i < cost.length; ++i) {
            p = q;
            q = r;
            r = Math.min(q + cost[i], p + cost[i - 1]);
        }
        return r;
    }


    //509. 斐波那契数
    //斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
    //F(0) = 0,   F(1) = 1
    //F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
    //给定 N，计算 F(N)。
    //
    //示例 1：
    //
    //输入：2
    //输出：1
    //解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
    //像动态规划，注意边界条件即可
    //时间复杂度：O(N)
    //空间复杂度：O(1)
    public int fib(int N) {
        int p = 0, q = 0, r = 1;
        if (N == 0) return 0;
        for (int i = 2; i <= N; ++i) {
            p = q;
            q = r;
            r = p + q;
        }
        return r;
    }

    //方法二：递归
    //时间复杂度：O(2^N)。这是计算斐波那契数最慢的方法。因为它需要指数的时间。====》因为重复计算了
    //空间复杂度：O(N)，在堆栈中我们需要与 N 成正比的空间大小。该堆栈跟踪 fib(N) 的函数调用，随着堆栈的不断增长如果没有足够的内存则会导致 StackOverflowError。
    public int fib2(int N) {
        if (N <= 1) {
            return N;
        }
        return fib2(N - 1) + fib2(N - 2);
    }

    //方法二：记忆化自底向上的方法
    //自底向上通过迭代计算斐波那契数的子问题并存储已计算的值，通过已计算的值进行计算。减少递归带来的重复计算。
    //
    //算法：
    //如果 N 小于等于 1，则返回 N。
    //迭代 N，将计算出的答案存储在数组中。
    //使用数组前面的两个斐波那契数计算当前的斐波那契数。
    //知道我们计算到 N，则返回它的斐波那契数。
    //时间复杂度：O(N)
    //空间复杂度：O(N)，使用了空间大小为 N 的数组。
    public int fib3(int N) {
        if (N <= 1) {
            return N;
        }
        return memoize(N);
    }

    public int memoize(int N) {
        int[] cache = new int[N + 1];
        cache[1] = 1;
        for (int i = 2; i <= N; i++) {
            cache[i] = cache[i - 1] + cache[i - 2];
        }
        return cache[N];
    }

    //方法三：记忆化自顶向下的方法
    //我们先计算存储子问题的答案，然后利用子问题的答案计算当前斐波那契数的答案。我们将递归计算，但是通过记忆化不重复计算已计算的值。
    //
    //算法：
    //
    //如果 N <= 1，则返回 N。
    //调用和返回 memoize(N)。
    //如果 N 对应的斐波那契数存在，则返回。
    //否则将计算 N 对应的斐波那契数为 memoize(N-1) + memoize(N-2)。
    //时间复杂度：O(N)
    //空间复杂度：O(N)，内存中使用的堆栈大小。
    private Integer[] cache = new Integer[31];

    public int fib4(int N) {
        if (N <= 1) {
            return N;
        }
        cache[0] = 0;
        cache[1] = 1;
        return memoize2(N);
    }

    public int memoize2(int N) {
        if (cache[N] != null) {
            return cache[N];
        }
        cache[N] = memoize2(N - 1) + memoize2(N - 2);
        return memoize(N);
    }





    //1137. 第 N 个泰波那契数
    //泰波那契序列 Tn 定义如下： 
    //
    //T0 = 0, T1 = 1, T2 = 1, 且在 n >= 0 的条件下 Tn+3 = Tn + Tn+1 + Tn+2
    //给你整数 n，请返回第 n 个泰波那契数 Tn 的值。
    //
    //示例 1：
    //输入：n = 4
    //输出：4
    //解释：
    //T_3 = 0 + 1 + 1 = 2
    //T_4 = 1 + 1 + 2 = 4
    public int tribonacci(int n) {
        int x = 0, y = 0, z = 1;
        int r = 1;
        if (n == 0) return 0;
        if (n <= 2) return 1;
        for (int i = 3; i <= n; i++) {
            x = y;
            y = z;
            z = r;
            r = x + y + z;
        }
        return r;
    }


    //面试题64. 求1+2+…+n
    //求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
    //方法1：递归
    public int sumNums(int n) {
        //如果可以用判断语句，很快就能想到用递归，但是不能用判断语句
        //return n == 0 ? 0 : n + sumNums(n - 1);
        int sum = n;
        boolean flag = n > 0 && (sum += sumNums(n - 1)) > 0; //巧妙利用"短路"特性 当n大于0时 就继续递归 否则停止递归 return 前面的累加值
        return sum;
    }

    //面试题29. 顺时针打印矩阵，见下一题54
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return new int[0];
        }
        int rows = matrix.length, columns = matrix[0].length;
        boolean[][] visited = new boolean[rows][columns];
        int total = rows * columns;
        int[] order = new int[total];
        int row = 0, column = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            order[i] = matrix[row][column];

            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0];
            int nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }

    //54. 螺旋矩阵
    //给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
    //方法一：可以模拟螺旋矩阵的路径。
    // 初始位置是矩阵的左上角，初始方向是向右，当路径超出界限或者进入之前访问过的位置时，则顺时针旋转，进入下一个方向。
    //判断路径是否进入之前访问过的位置需要使用一个与输入矩阵大小相同的辅助矩阵visited，其中的每个元素表示该位置是否被访问过。当一个元素被访问时，将visited 中的对应位置的元素设为已访问。
    //如何判断路径是否结束？由于矩阵中的每个元素都被访问一次，因此路径的长度即为矩阵中的元素数量，当路径的长度达到矩阵中的元素数量时即为完整路径，将该路径返回。
    //时间复杂度：O(mn)，其中 m 和 n 分别是输入矩阵的行数和列数。矩阵中的每个元素都要被访问一次。
    //空间复杂度：O(mn)。需要创建一个大小为 m×n 的矩阵visited 记录每个位置是否被访问过。
    public List<Integer> spiralOrder_54(int[][] matrix) {
        List<Integer> order = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return order;
        }
        int rows = matrix.length, columns = matrix[0].length;
        boolean[][] visited = new boolean[rows][columns];
        int total = rows * columns;
        int row = 0, column = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            order.add(matrix[row][column]);
            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0];
            int nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }

    //方法二：按照层级来，见官网


    //59. 螺旋矩阵 II
//    public int[][] generateMatrix(int n) {
//
//        return {{0,0}};
//    }




}
