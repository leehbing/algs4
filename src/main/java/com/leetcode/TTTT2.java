package com.leetcode;


import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 20/5/2020 3:43 PM
 * @Author: lihongbing
 */
public class TTTT2 {
    //394. 字符串解码        给定一个经过编码的字符串，返回它解码后的字符串。
    //s = "3[a]2[bc]", 返回 "aaabcbc".
    //s = "3[a2[c]]", 返回 "accaccacc".
    //s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
    //
    //我的方法，这个和利用栈计算表达式一样的，参考edu.princeton.cs.algs4.Evaluate
    public static String decodeString(String s) {
        //操作数栈
        Stack<String> ops = new Stack<String>();
        //运算符栈
        Stack<Character> vals = new Stack<Character>();
        //先考虑数字都是个位数
        for (int i = 0; i < s.length(); i++) {


        }


        return null;
    }


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


}
