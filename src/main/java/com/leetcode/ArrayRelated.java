package com.leetcode;


import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 9/6/2020 11:24 AM
 * @Author: lihongbing
 */
public class ArrayRelated {
    //35. 搜索插入位置
    //给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
    //你可以假设数组中无重复元素。
    //示例 1:
    //输入: [1,3,5,6], 5
    //输出: 2
    //示例 2:
    //输入: [1,3,5,6], 2
    //输出: 1
    //示例 3:
    //输入: [1,3,5,6], 7
    //输出: 4
    //遍历
    // 时间复杂度：O(n)
    public int searchInsert(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < target) {
                continue;
            } else if (nums[i] == target) {
                return i;
            } else {
                return i;

            }
        }
        return nums.length;
    }

    //方法二：二分查找
    //时间复杂度：O(logn)
    public int searchInsert2(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    //278. 第一个错误的版本
    //你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
    //假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
    //你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。
    //
    //示例:
    //给定 n = 5，并且 version = 4 是第一个错误的版本。
    //
    //调用 isBadVersion(3) -> false
    //调用 isBadVersion(5) -> true
    //调用 isBadVersion(4) -> true
    //
    //所以，4 是第一个错误的版本。 
    //二分查找第一个错误的版本
    //left指向的都是正确的版本，right指向的都是错误的版本，left和right越来越接近，最终left+1=right
    ////时间复杂度：O(logn)
    public static int firstBadVersion(int n) {
        int left = 1;
        int right = n;
        if (isBadVersion(left)) return left;
        while ((right - left) != 1) {
            //int mid =(left + right) / 2; //针对测试用例2126753390 1702766719原因是这里溢出了
            Long tmp = (Long.valueOf(left) + Long.valueOf(right)) / 2;
            int mid = tmp.intValue();

            if (isBadVersion(mid)) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return right;
    }

    public static boolean isBadVersion(int version) {
        return version >= 1702766719 ? true : false;
    }

    //374. 猜数字大小
    //我们正在玩一个猜数字游戏。 游戏规则如下：
    //我从 1 到 n 选择一个数字。 你需要猜我选择了哪个数字。
    //每次你猜错了，我会告诉你这个数字是大了还是小了。
    //你调用一个预先定义好的接口 guess(int num)，它会返回 3 个可能的结果（-1，1 或 0）：
    //
    //-1 : 我的数字比较小
    // 1 : 我的数字比较大
    // 0 : 恭喜！你猜对了！
    //
    //示例 :
    //输入: n = 10, pick = 6
    //输出: 6
    //    //时间复杂度：O(logn)
    public static int guessNumber(int n) {
        int left = 1;
        int right = n;
        while (left <= right) {
            //int mid =(left + right) / 2;
            Long tmp = (Long.valueOf(left) + Long.valueOf(right)) / 2;
            int mid = tmp.intValue();

            if (guess(mid) == 0) {
                return mid;
            } else if (guess(mid) == -1) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    static int  guess(int num) {
        if(num==6){
            return 0;
        }else if(num>6){
            return -1;
        }else{
            return 1;
        }
    }


    public static void main(String[] args) {
//        System.out.println(firstBadVersion(2126753390));
        System.out.println(guessNumber(10));
    }


    //66. 加一
    //给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
    //
    //最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
    //
    //你可以假设除了整数 0 之外，这个整数不会以零开头。
    public int[] plusOne(int[] digits) {
        List<Integer> temp = new ArrayList<>();
        int carry = 1;
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] + carry >= 10) {
                temp.add((digits[i] + carry) % 10);
                carry = (digits[i] + carry) / 10;
            } else {
                temp.add(digits[i] + carry);
                carry = 0;
            }
        }
        if (carry == 1) temp.add(carry);
        Collections.reverse(temp);
//        return temp.toArray(new Integer[0]);
        return temp.stream().mapToInt(i -> i).toArray();

    }

    //349. 两个数组的交集
    //给定两个数组，编写一个函数来计算它们的交集。
    //输入：nums1 = [1,2,2,1], nums2 = [2,2]
    //输出：[2]
    //方法一：两个hashset
    //时间复杂度：O(m+n)
    //空间复杂度：O(m+n)
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> result = new HashSet<>();
        for (int num : nums1) {
            set1.add(num);
        }
        for (int num : nums2) {
            if (set1.contains(num)) {
                result.add(num);
            }
        }
        int[] ans = new int[result.size()];
        int i = 0;
        for (int num : result) {
            ans[i++] = num;
        }
        return ans;
    }

    //350. 两个数组的交集 II
    //给定两个数组，编写一个函数来计算它们的交集。
    //示例 1:
    //输入: nums1 = [1,2,2,1], nums2 = [2,2]
    //输出: [2,2]
    //方法一：两个hashmap
    //时间复杂度：O(n+m)。其中 n，m 分别代表了数组的大小。
    //空间复杂度：O(min(n,m))，我们对较小的数组进行哈希映射使用的空间。
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map1 = new HashMap<>();
        List<Integer> result = new ArrayList<>();
        for (int num : nums1) {
            if (map1.containsKey(num)) {
                map1.put(num, map1.get(num) + 1);
            } else {
                map1.put(num, 1);
            }
        }
        for (int num : nums2) {
            if (map1.containsKey(num) && map1.get(num) > 0) {
                result.add(num);
                map1.put(num, map1.get(num) - 1);
            }

        }
        int[] ans = new int[result.size()];
        int i = 0;
        for (int num : result) {
            ans[i++] = num;
        }
        return ans;
    }

    //方法二：排序
    //当输入数据是有序的，推荐使用此方法。在这里，我们对两个数组进行排序，并且使用两个指针在一次扫面找出公共的数字。
    //算法：
    //
    //对数组 nums1 和 nums2 排序。
    //初始化指针 i，j 和 k 为 0。
    //指针 i 指向 nums1，指针 j 指向 nums2：
    //如果 nums1[i] < nums2[j]，则 i++。
    //如果 nums1[i] > nums2[j]，则 j++。
    //如果 nums1[i] == nums2[j]，将元素拷贝到 nums1[k]，且 i++，j++，k++。
    //返回数组 nums1 前 k 个元素。

    //时间复杂度：O(nlogn+mlogm)。其中 n，m 分别代表了数组的大小。我们对数组进行了排序然后进行了线性扫描。
    //空间复杂度：O(1)，我们忽略存储答案所使用的空间，因为它对算法本身并不重要。
    public int[] intersect2(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0, j = 0, k = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                ++i;
            } else if (nums1[i] > nums2[j]) {
                ++j;
            } else {
                nums1[k++] = nums1[i++];
                ++j;
            }
        }
        return Arrays.copyOfRange(nums1, 0, k);
    }

    //1002. 查找常用字符
    //给定仅有小写字母组成的字符串数组 A，返回列表中的每个字符串中都显示的全部字符（包括重复字符）组成的列表。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。
    //你可以按任意顺序返回答案。
    //
    //示例 1：
    //输入：["bella","label","roller"]
    //输出：["e","l","l"]
    //示例 2：
    //输入：["cool","lock","cook"]
    //输出：["c","o"]
    //时间复杂度：O(n*m)
    //空间复杂度：O(100*26)
    public List<String> commonChars(String[] A) {
        List<String> ans = new ArrayList<>();
        int[][] num = new int[100][26];
        for (int i = 0; i < A.length; i++)   //建立一个二维数组，标记所有出现的字母次数
            for (int j = 0; j < A[i].length(); j++)
                num[i][(A[i].charAt(j) - 'a')]++;
        for (int j = 0; j < 26; j++)                  //将所有列的最小值存到第一行,这样子第一行的数字即表示公共部分
            for (int i = 1; i < A.length; i++)
                num[0][j] = Math.min(num[0][j], num[i][j]);
        //按照第一行保存的次数输出相应字母
        for (int i = 0; i < 26; i++) {
            while (num[0][i] > 0) {
                ans.add(String.valueOf((char) ('a' + i)));
                num[0][i]--;
            }
        }
        return ans;

    }


}
