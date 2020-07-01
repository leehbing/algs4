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

    static int guess(int num) {
        if (num == 6) {
            return 0;
        } else if (num > 6) {
            return -1;
        } else {
            return 1;
        }
    }


    //375. 猜数字大小 II
    //我们正在玩一个猜数游戏，游戏规则如下：
    //
    //我从 1 到 n 之间选择一个数字，你来猜我选了哪个数字。
    //
    //每次你猜错了，我都会告诉你，我选的数字比你的大了或者小了。
    //
    //然而，当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。直到你猜到我选的数字，你才算赢得了这个游戏。
    //
    //示例:
    //
    //n = 10, 我选择了8.
    //
    //第一轮: 你猜我选择的数字是5，我会告诉你，我的数字更大一些，然后你需要支付5块。
    //第二轮: 你猜是7，我告诉你，我的数字更大一些，你支付7块。
    //第三轮: 你猜是9，我告诉你，我的数字更小一些，你支付9块。
    //
    //游戏结束。8 就是我选的数字。
    //
    //你最终要支付 5 + 7 + 9 = 21 块钱。
    //给定 n ≥ 1，计算你至少需要拥有多少现金才能确保你能赢得这个游戏。
    //
    //动态规划+二分查找====》太复杂度了，没看
    public int getMoneyAmount(int n) {
        return 0;
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


    //56. 合并区间
    //如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。
    //时间复杂度：O(nlogn)，其中 n 为区间的数量。除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 O(nlogn)。
    //空间复杂度：O(logn)，其中 n 为区间的数量。这里计算的是存储答案之外，使用的额外空间。O(logn) 即为排序所需要的空间复杂度。
    //
    public static int[][] merge(int[][] intervals) {
        // 先按照区间起始位置排序，下面方法调用的是归并排序
        Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
        // 遍历区间
        int[][] res = new int[intervals.length][2];
        int idx = -1;
        for (int[] interval : intervals) {
            // 如果结果数组是空的，或者当前区间的起始位置 > 结果数组中最后区间的终止位置，
            // 则不合并，直接将当前区间加入结果数组。
            if (idx == -1 || interval[0] > res[idx][1]) {
                res[++idx] = interval;
            } else {
                // 反之将当前区间合并至结果数组的最后区间
                res[idx][1] = Math.max(res[idx][1], interval[1]);
            }
        }
        return Arrays.copyOf(res, idx + 1);
    }

    //986. 区间列表的交集
    //给定两个由一些闭区间组成的列表，每个区间列表都是成对不相交的，并且已经排序。
    //返回这两个区间列表的交集。
    //时间复杂度：O(M+N)，其中 M, N 分别是数组 A 和 B 的长度。
    //空间复杂度：O(M+N)，答案中区间数量的上限。
    //
    public int[][] intervalIntersection(int[][] A, int[][] B) {
        List<int[]> ans = new ArrayList();
        int i = 0, j = 0;

        while (i < A.length && j < B.length) {
            // Let's check if A[i] intersects B[j].
            // lo - the startpoint of the intersection
            // hi - the endpoint of the intersection
            int lo = Math.max(A[i][0], B[j][0]);
            int hi = Math.min(A[i][1], B[j][1]);
            if (lo <= hi)
                ans.add(new int[]{lo, hi});

            // Remove the interval with the smallest endpoint
            if (A[i][1] < B[j][1])
                i++;
            else
                j++;
        }

        return ans.toArray(new int[ans.size()][]);
    }


    //215. 数组中的第K个最大元素
    //在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
    //示例 1:
    //输入: [3,2,1,5,6,4] 和 k = 2
    //输出: 5
    public static int findKthLargest(int[] nums, int k) {
        //思路一，降序排序
//        Arrays.sort(nums);
//        return nums[nums.length-k];
//        return 0;

        //思路二，优先队列，类似于找到TopK
        //        官方提供的优先队列是最小值在最顶点（PriorityQueue默认就是最小堆，堆，小顶堆），正好符合需要，否则需要自己定义比较器Comparator
        //        时间复杂度：O(nlgk)
        //        空间复杂度：O(k)

        Queue<Integer> integerPriorityQueue = new java.util.PriorityQueue<>();
        for (int i : nums) {
            integerPriorityQueue.add(i);
            if (integerPriorityQueue.size() > k) {
                integerPriorityQueue.remove();
            }
        }
        return integerPriorityQueue.peek();
    }

    //414. 第三大的数
    //给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。
    public static int thirdMax(int[] nums) {
        //思想一，先去重，再用优先队列，比较复杂

        //思想三，网友还有一个解答，用红黑树，等我看懂了红黑树再回来看

        //思想二，用三个变量来存放第一大，第二大，第三大的元素的变量，分别为one, two, three；
        //遍历数组，若该元素比one大则往后顺移一个元素，比two大则往往后顺移一个元素，比three大则赋值个three；
        //最后得到第三大的元素，若没有则返回第一大的元素。
        long MIN = Long.MIN_VALUE;    // MIN代表没有在值

        if (nums == null || nums.length == 0) throw new RuntimeException("nums is null or length of 0");
        int n = nums.length;

        int one = nums[0];
        long two = MIN;
        long three = MIN;

        for (int i = 1; i < n; i++) {
            int now = nums[i];
            if (now == one || now == two || now == three) continue;    // 如果存在过就跳过不看
            if (now > one) {
                three = two;
                two = one;
                one = now;
            } else if (now > two) {
                three = two;
                two = now;
            } else if (now > three) {
                three = now;
            }
        }

        if (three == MIN) return one;  // 没有第三大的元素，就返回最大值
        return (int) three;
    }

    //优先队列(堆，最小堆)的容量为k，最顶的元素就是我们要的第k大的元素，如果新加的元素小于等于peek(),忽略，否则移除最顶端元素，加进新元素，再返回最顶端元素
    //时间复杂度：O(nlgk)
    //空间复杂度：O(k)
    class KthLargest {
        PriorityQueue<Integer> priorityQueue = null;
        int k;

        public KthLargest(int k, int[] nums) {
            this.k = k;
            priorityQueue = new PriorityQueue<>(k);
            for (int num : nums) {
                priorityQueue.add(num);
                if (priorityQueue.size() > k) {
                    priorityQueue.remove();
                }
            }
        }

        public int add(int val) {
            if (priorityQueue.size() < k) {
                priorityQueue.offer(val);

            } else if (val > priorityQueue.peek()) {
                priorityQueue.remove();
                priorityQueue.add(val);
            }
            return priorityQueue.peek();
        }
    }

    //347. 前 K 个高频元素        给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
    //优先队列(最小堆)的使用
    //时间复杂度：O(nlgk)
    //空间复杂度：O(n)
    public static int[] topKFrequent(int[] nums, int k) {

        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        PriorityQueue<Integer> priorityQueue = new PriorityQueue<Integer>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer key1, Integer key2) {
                return map.get(key1).compareTo(map.get(key2));
//                return ((Comparable<String>) map.get(key1)).compareTo(map.get(key2);
            }
        });

        for (int n : map.keySet()) {
            priorityQueue.add(n);
            if (priorityQueue.size() > k) {
                priorityQueue.remove();
            }
        }
        // build output list
        List<Integer> top_k = new ArrayList<>();
        while (!priorityQueue.isEmpty())
            top_k.add(priorityQueue.poll());
        Collections.reverse(top_k); //倒序
        int[] result = new int[k];
        for (int i = 0; i < top_k.size(); i++) {
            result[i] = top_k.get(i);
        }
        return result;
    }

    //324. 摆动排序 II
    //给定一个无序的数组 nums，将它重新排列成 nums[0] < nums[1] > nums[2] < nums[3]... 的顺序。
    //
    //示例 1:
    //
    //输入: nums = [1, 5, 1, 1, 6, 4]
    //输出: 一个可能的答案是 [1, 4, 1, 5, 1, 6]
    //示例 2:
    //
    //输入: nums = [1, 3, 2, 2, 3, 1]
    //输出: 一个可能的答案是 [2, 3, 1, 3, 1, 2]
    public void wiggleSort(int[] nums) {
        //思想一：先快速排序，然后从中间split开来，交叉
        //时间复杂度，主要是快速排序的，O(nlgn)
        //空间复杂度： O(nlgn)
        Arrays.sort(nums);
        int length = nums.length;
        int smallerLength = length % 2 == 1 ? length / 2 + 1 : length / 2;
        int[] smaller = new int[smallerLength];
        int[] bigger = new int[length / 2];

//        Arrays.copyOf(nums,newLength);
// Arrays.copyOf()只能从起始位置复制newLength长度的元素,这个方法底层用的是System.arraycopy

        System.arraycopy(nums, 0, smaller, 0, smallerLength);
        System.arraycopy(nums, smallerLength, bigger, 0, length / 2);


        for (int i = 0; i < length / 2; i++) {
            //交叉，为了避免[4 5 5 6]的问题，将子数组反向使用
            nums[2 * i] = smaller[smallerLength - 1 - i];
            nums[2 * i + 1] = bigger[bigger.length - 1 - i];
        }
        if (length % 2 == 1) {
            nums[length - 1] = smaller[0];
        }
        //
        //https://leetcode-cn.com/problems/wiggle-sort-ii/solution/yi-bu-yi-bu-jiang-shi-jian-fu-za-du-cong-onlognjia/
        //2. 解法2：快速选择 + 3-way-partition
        //3. 解法3：快速选择 + 3-way-partition + 虚地址
    }


    //27. 移除元素    仅使用 O(1) 额外空间
    //给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
    //不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
    //元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
    //示例 1:
    //给定 nums = [3,2,2,3], val = 3,
    //函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
    //你不需要考虑数组中超出新长度后面的元素
    //
    //方法一：双指针     快指针j，慢指针i
    //当 nums[j]nums[j] 与给定的值相等时，递增 jj 以跳过该元素。只要 nums[j] != val,我们就复制 nums[j] 到 nums[i] 并同时递增两个索引。重复这一过程，直到 j 到达数组的末尾，该数组的新长度为 i。
    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                nums[i] = nums[j];
                i++;
            }
        }
        return i;
    }

    //方法二：双指针 —— 当要删除的元素很少时
    //现在考虑数组包含很少的要删除的元素的情况。例如，num=[1，2，3，5，4]，Val=4 之前的算法会对前四个元素做不必要的复制操作。
    // 另一个例子是 num=[4，1，2，3，5]，Val=4。似乎没有必要将 [1，2，3，5] 这几个元素左移一步，因为问题描述中提到元素的顺序可以更改。
    //当我们遇到 nums[i] = val 时，我们可以将当前元素与最后一个元素进行交换，并释放最后一个元素。这实际上使数组的大小减少了 1。
    //请注意，被交换的最后一个元素可能是您想要移除的值。但是不要担心，在下一次迭代中，我们仍然会检查这个元素。
    public int removeElement2(int[] nums, int val) {
        int i = 0;
        int n = nums.length;
        while (i < n) {
            if (nums[i] == val) {
                nums[i] = nums[n - 1];
                // reduce array size by one
                n--;
            } else {
                i++;
            }
        }
        return n;
    }

    //26. 删除排序数组中的重复项
    //给定 nums = [0,0,1,1,1,2,2,3,3,4],
    //函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
    //你不需要考虑数组中超出新长度后面的元素。
    //解法同27题，快慢指针
    public static int removeDuplicates(int[] nums) {
        if (nums.length == 0) return 0;
        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    //80. 删除排序数组中的重复项 II
    //给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
    //给定 nums = [0,0,1,1,1,1,2,3,3],
    //函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。
    //你不需要考虑数组中超出新长度后面的元素
    //快慢指针
    //或者增加一个计数器count来记录重复数组的个数
    public static int removeDuplicates_80(int[] nums) {
        if (nums.length < 3) return nums.length;
        int i = 1;
        for (int j = 2; j < nums.length; j++) {
            if (nums[j] != nums[i - 1]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    //560. 和为K的子数组  给定一个整数数组和一个整数 k，你需要找到该数组中和为 k 的连续的子数组的个数。
    //方法一：暴力法
    //时间复杂度：O(n^2)
    //空间复杂度：O(1)
    public static int subarraySum(int[] nums, int k) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
        //上面的方法换一种写法，
        //  2   4   3   8   3   6   4   8   2   4   3   5
        //         end            start
        //          j               i
        //int count = 0;
        //for (int start = 0; start < nums.length; ++start) {
        //    int sum = 0;
        //    for (int end = start; end >= 0; --end) {
        //        sum += nums[end];
        //        if (sum == k) {
        //            count++;
        //        }
        //    }
        //}
        //return count;
    }

    //引入前缀和：定义 pre[i] 为 [0..i] 里所有数的和，题目等价转化：
    //从【有几种 i、j 组合，使得从第 i 到 j 项的子数组的求和 === k】
    //↓ ↓ ↓ 转化为 ↓ ↓ ↓
    //【有几种 i、j 组合，满足 i < j 且 prefixSum[ j ] - prefixSum[ i - 1 ] === k】
    //于是我们想求出 prefixSum 数组的每一项，再看哪些项相减 === k，统计 count
    //但通式有 i、j 2 个变量，需要两层 for 循环，时间复杂度依旧是 O(n^2)
    //
    //摈弃 prefixSum 数组，引入哈希表
    //可以不用 prefixSum 数组吗？可以。
    //因为我们不关心 前缀和 具体对应哪一项，只关心 前缀和 的值和 出现次数。
    //用 prefixSum 变量，保存当前项的前缀和，存入 map
    //这样 map 代替了 prefixSum 数组，记录出现过的 前缀和 和 出现次数


    //方法二：前缀和 + 哈希表优化
    // pre[i] 可以由pre[i−1] 递推而来，即：pre[i]=pre[i−1]+nums[i]
    //考虑以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为pre[i]−k 的 pre[j] （符合条件的子数组的位置为j+1,j+2....i）即可。
    //我们建立哈希表 mp，以和为键，出现次数为对应的值，记录 pre[i] 出现的次数，从左往右边更新 mp 边计算答案，
    //那么以 i 结尾的答案 mp[pre[i]−k] 即可在 O(1)时间内得到。最后的答案即为所有下标结尾的和为 k 的子数组个数之和。
    // i     空  0   1   2   3   4   5   6   7   8   9
    //pre[i] 0   1   0   0   -1  0   1   2   4   5   6
    //       1   1  -1   0   -1  1   1   1   2   1   1         k=2
    //                                   i
    //                                 pre[6]-k=0
    //              则j=空，1，2，4都是符合的
    //时间复杂度：O(n)，其中 n 为数组的长度。我们遍历数组的时间复杂度为 O(n)，中间利用哈希表查询删除的复杂度均为O(1)，因此总时间复杂度为 O(n)。
    //空间复杂度：O(n)。哈希表在最坏情况下可能有 n 个不同的键值，因此需要 O(n) 的空间复杂度
    public static int subarraySum2(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        mp.put(0, 1);//对于一开始的情况，下标 0 之前没有元素，可以认为前缀和为 0，个数为 1 个
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (mp.containsKey(pre - k)) //j+1,j+2,...i，所以应该是先判断，然后在put后面的前缀和
                count += mp.get(pre - k);
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;
    }

    //1248. 统计「优美子数组」
    //给你一个整数数组 nums 和一个整数 k。
    //如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
    //请返回这个数组中「优美子数组」的数目。
    //  本题更本质其实就是这样的 奇数为1, 偶数为0, 子区间和为k的种类数，   相同题 -> 560. 和为K的子数组
    public static int numberOfSubarrays(int[] nums, int k) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (nums[i] % 2 == 1) {
                nums[i] = 1; //奇数
            } else {
                nums[i] = 0;
            }
        }
        int count = 0, pre = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        mp.put(0, 1);//前缀和+哈希，见题目560
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (mp.containsKey(pre - k))
                count += mp.get(pre - k);
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;

    }

    //523. 连续的子数组和
    //给定一个包含非负数的数组和一个目标整数 k，
    //编写一个函数来判断该数组是否含有连续的子数组，其大小至少为2，总和为 k 的倍数，即总和为 n*k，其中 n 也是一个整数。
    public static boolean checkSubarraySum(int[] nums, int k) {
        int count = 0, pre = nums[0]; //下面拿前缀和修改了一下，但是提交后报超时，可能因为时间复杂度是O(n)
        int prepre = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        mp.put(0, 1);//对于一开始的情况，下标 0 之前没有元素，可以认为前缀和为 0，个数为 1 个
        for (int i = 1; i < nums.length; i++) {
            prepre = pre;
            pre += nums[i];
//            if (mp.containsKey(pre - k))
//                count += mp.get(pre - k);
            //遍历map
            for (int key : mp.keySet()) {
                if ((k == 0 && (pre - key) == 0) || (k != 0 && (pre - key) % k == 0)) {
                    count += mp.get(key); //这句话改成return true就不报超时了，可能如果是count += mp.get(key)，时间复杂度会变成O(n^2)力扣后台的测试用例不通过
                }
            }
            mp.put(prepre, mp.getOrDefault(prepre, 0) + 1);
        }
        return count > 0;
    }

    //下面是官网的解答，也是基于前缀和，但是改的很优雅
    //首先，取余%运算支持分配律，结合律，比如(a+b)%k=a%k+b%k,可以这么想，有整数k的部分先减掉，不影响后面的取余运算===》同余定理
    public static boolean checkSubarraySum2(int[] nums, int k) {
        int sum = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);//(sum,索引i)
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (k != 0)
                sum = sum % k; //比如k=99，假设此时的sum=20=pre[i]
            if (map.containsKey(sum)) { //如果之前的存在20，这样pre[i]-20=0, 0%99=0，正好整除
                if (i - map.get(sum) > 1) //存在也不更新key对应的value，这样保证i - map.get(sum)会最大，也有可能成功
                    return true;
            } else {
                map.put(sum, i); //如果不存在20，则加进去(20 -> i)
            }
        }
        return false;
    }

    //974. 和可被 K 整除的子数组
    //给定一个整数数组 A，返回其中元素之和可被 K 整除的（连续、非空）子数组的数目。
    //方法一
    public static int subarraysDivByK(int[] A, int K) {
        //我的方法，前缀和
        //时间复杂度：O(n^2), 提交上去会报超出时间限制，应该是时间复杂度太大了
        //空间复杂度: O(n)
        int[] sum = new int[A.length];
        int temp = 0;
        int count = 0;
        for (int i = 0; i < A.length; i++) {
            temp += A[i];
            sum[i] = temp;
        }
        for (int i = A.length - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) { //j=i
                if ((sum[i] - sum[j] + A[j]) % K == 0) count++;
            }

        }
        return count;
    }

    //方法二
    public static int subarraysDivByK2(int[] A, int K) {
        //方法二，前缀和+hashMap
        //时间复杂度：O(N)
        //空间复杂度：O(min(N,K))，即hash表需要的空间，当 N≤K 时，最多有 N 个前缀和，因此哈希表中最多有 N+1个键值对；
        // 当 N>K 时，最多有 K 个不同的余数，因此哈希表中最多有 K 个键值对。也就是说，哈希表需要的空间取决于 N 和 K 中的较小值。
        //
        //

        int sum = 0;
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);//对于一开始的情况，下标 0 之前没有元素，可以认为前缀和为 0，个数为 1 个, 考虑了前缀和本身被 K 整除的情况
        for (int i = 0; i < A.length; i++) {
            sum += A[i];
            // sum = sum % K;
            //// !!!!注意 Java 取模的特殊性，当被除数为负数时取模结果为负数，需要纠正
            sum = (sum % K + K) % K;

            if (map.containsKey(sum)) {
                count += map.get(sum);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);

        }
        return count;
    }

    //724. 寻找数组的中心索引
    //给定一个整数类型的数组 nums，请编写一个能够返回数组 “中心索引” 的方法。
    //
    //我们是这样定义数组 中心索引 的：数组中心索引的左侧所有元素相加的和等于右侧所有元素相加的和。
    //
    //如果数组不存在中心索引，那么我们应该返回 -1。如果数组有多个中心索引，那么我们应该返回最靠近左边的那一个。
    public static int pivotIndex(int[] nums) {
        //我的解法：前缀和+后缀和
        //时间复杂度：O(n)
        //空间复杂度：O(n)
        int[] sumpre = new int[nums.length];
        int sum1 = 0;
        int[] sumpost = new int[nums.length];
        int sum2 = 0;
        for (int i = 0; i < nums.length; i++) {
            sum1 += nums[i];
            sumpre[i] = sum1;
            sum2 += nums[nums.length - i - 1];
            sumpost[nums.length - i - 1] = sum2;
        }
        for (int i = 0; i < nums.length; i++) {
            if (sumpre[i] == sumpost[i]) return i;
        }
        return -1;
    }

    public int pivotIndex2(int[] nums) { //更简单的解法。。。
        int sum = 0, leftsum = 0;
        for (int x : nums) sum += x;
        for (int i = 0; i < nums.length; ++i) {
            if (leftsum == sum - leftsum - nums[i]) return i;
            leftsum += nums[i];
        }
        return -1;
    }


    //53. 最大子序和
    //给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
    //
    //示例:
    //
    //输入: [-2,1,-3,4,-1,2,1,-5,4],
    //输出: 6
    //解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
    //进阶: 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
    //方法一：动态规划
    //思路和算法
    //
    //假设 nums 数组的长度是 n，下标从 0 到 n-1。
    //
    //我们用 ai代表 nums[i]，用f(i) 代表以第 i 个数结尾的「连续子数组的最大和」，那么很显然我们要求的答案就是：
    //  max{f(i)}   0≤i≤n−1
    //
    //因此我们只需要求出每个位置的 f(i)，然后返回 f 数组中的最大值即可。那么我们如何求 f(i) 呢？我们可以考虑 ai 单独成为一段还是加入f(i−1) 对应的那一段，
    // 这取决于 ai和 ff(i−1)+ai的大小，我们希望获得一个比较大的，于是可以写出这样的动态规划转移方程：
    //  f(i)=max{f(i−1)+ai,ai}
    //不难给出一个时间复杂度 O(n)、空间复杂度 O(n)的实现，即用一个 f 数组来保存 f(i) 的值，用一个循环求出所有 f(i)。
    // 考虑到 f(i)只和 f(i - 1)相关，于是我们可以只用一个变量 pre 来维护对于当前 f(i) 的 f(i−1) 的值是多少，从而让空间复杂度降低到 O(1)，这有点类似「滚动数组」的思想。
    //
    //
    //时间复杂度：O(n)，其中 n 为 nums 数组的长度。我们只需要遍历一遍数组即可求得答案。
    //空间复杂度：O(1)。我们只需要常数空间存放若干变量。
    //
    public int maxSubArray(int[] nums) {
        int pre = 0, maxAns = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            maxAns = Math.max(maxAns, pre);
        }
        return maxAns;
    }


    //152. 乘积最大子数组
    //给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
    //示例 1:
    //输入: [2,3,-2,4]
    //输出: 6
    //解释: 子数组 [2,3] 有最大乘积 6。
    //方法一：动态规划
    //思路和算法
    //
    //如果我们用 fmax(i) 开表示以第 i 个元素结尾的乘积最大子数组的乘积，a 表示输入参数 nums，
    // 那么根据「53. 最大子序和」的经验，我们很容易推导出这样的状态转移方程：
    //      max{f(i−1)×ai,ai}    1<=i<=n
    //它表示以第 i 个元素结尾的乘积最大子数组的乘积可以考虑 ai加入前面的 fmax(i−1) 对应的一段，或者单独成为一段，这里两种情况下取最大值。求出所有的fmax(i) 之后选取最大的一个作为答案。
    //
    //可是在这里，这样做是错误的。为什么呢？？？？？
    //
    //因为这里的定义并不满足「最优子结构」。具体地讲，如果 a={5,6,−3,4,−3}，那么此时fmax对应的序列是 {5,30,−3,4,−3}，
    // 按照前面的算法我们可以得到答案为 30，即前两个数的乘积，而实际上答案应该是全体数字的乘积。我们来想一想问题出在哪里呢？
    // 问题出在最后一个 −3 所对应的 fmax的值既不是 −3，也不是 4×−3，而是 5×30×(−3)×4×(−3)。
    // 所以我们得到了一个结论：当前位置的最优解未必是由前一个位置的最优解转移得到的。
    //
    //我们可以根据正负性进行分类讨论。
    //
    //考虑当前位置如果是一个负数的话，那么我们希望以它前一个位置结尾的某个段的积也是个负数，这样就可以负负得正，并且我们希望这个积尽可能「负得更多」，即尽可能小。
    //如果当前位置是一个正数的话，我们更希望以它前一个位置结尾的某个段的积也是个正数，并且希望它尽可能地大。
    // 于是这里我们可以再维护一个fmin(i)，它表示以第 i 个元素结尾的乘积最小子数组的乘积，那么我们可以得到这样的动态规划转移方程：
    //      fmax(i)=max{fmax(i-1)xai,fmin(i-1)xai,ai}
    //      fmin(i)=min{fmax(i-1)xai,fmin(i-1)xai,ai}
    //
    //它代表第 i 个元素结尾的乘积最大子数组的乘积 fmax(i)，可以考虑把 ai加入第 i−1 个元素结尾的乘积最大或最小的子数组的乘积中，二者加上ai，三者取大，
    // 就是第 i 个元素结尾的乘积最大子数组的乘积。
    // 第 i 个元素结尾的乘积最小子数组的乘积fmin(i) 同理。
    //易得这里的渐进时间复杂度和渐进空间复杂度都是 O(n)。
    //
    // 考虑优化空间。由于第 i 个状态只和第 i−1 个状态相关，根据「滚动数组」思想，我们可以只用两个变量来维护 i−1 时刻的状态，一个维护fmax ，一个维护fmin
    //​

    public static int maxProduct(int[] nums) {
        int[] maxF = new int[nums.length];
        int[] minF = new int[nums.length];
        maxF[0] = nums[0];
        minF[0] = nums[0];
        int ans = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            maxF[i] = Math.max(maxF[i - 1] * nums[i], Math.max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = Math.min(minF[i - 1] * nums[i], Math.min(nums[i], maxF[i - 1] * nums[i]));
            ans = Math.max(ans, maxF[i]);
        }
        return ans;
    }

    //方法二，暴力法
    //时间复杂度：O(n^2 + n)
    //空间复杂度：O(n)
    public static int maxProduct2(int[] nums) {
        int[] maxs = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            int temp = 1;
            int max = Integer.MIN_VALUE;
            for (int j = i; j < nums.length; j++) {
                temp *= nums[j];
                if (temp > max) max = temp;
            }
            maxs[i] = max;
        }

        int result = Integer.MIN_VALUE;
        ;
        for (int m = 0; m < maxs.length; m++) {
            if (result < maxs[m]) result = maxs[m];
        }
        return result;
    }

    //1295. 统计位数为偶数的数字
    //给你一个整数数组 nums，请你返回其中位数为 偶数 的数字的个数。
    //示例 1：
    //输入：nums = [12,345,2,6,7896]
    //输出：2
    //解释：
    //12 是 2 位数字（位数为偶数） 
    //345 是 3 位数字（位数为奇数）  
    //2 是 1 位数字（位数为奇数） 
    //6 是 1 位数字 位数为奇数） 
    //7896 是 4 位数字（位数为偶数）  
    //因此只有 12 和 7896 是位数为偶数的数字
    //
    //方法一：将每一个数子转换为字符串，然后判断字符串中含有的字符个数
    //时间复杂度：O(n)
    public static int findNumbers(int[] nums) {
        int count = 0;
        for (int num : nums) {
            if (String.valueOf(num).length() % 2 == 0) count++;
        }
        return count;
    }

    //方法二：
    //我们也可以使用语言内置的以 10 为底的对数函数 log10() 来得到整数 x 包含的数字个数。
    //一个包含 k 个数字的整数 x 满足不等式 10^(k-1) <= x < 10^k。
    // 将不等式取对数，得到 k - 1 <= log10(x) < k，
    // 因此我们可以用 k = floor(log10(x) + 1)得到 x 包含的数字个数 k，例如 floor(5.2) = 5
    //时间复杂度：O(n)
    public static int findNumbers2(int[] nums) {
        int count = 0;
        for (int num : nums) {
            if (Math.floor(Math.log10(num) + 1) % 2 == 0) count++;
        }
        return count;
    }

    //599. 两个列表的最小索引总和
    //假设Andy和Doris想在晚餐时选择一家餐厅，并且他们都有一个表示最喜爱餐厅的列表，每个餐厅的名字用字符串表示。
    //
    //你需要帮助他们用最少的索引和找出他们共同喜爱的餐厅。 如果答案不止一个，则输出所有答案并且不考虑顺序。 你可以假设总是存在一个答案。
    //方法一：暴力法
    //时间复杂度：O(m+n)
    //空间复杂度：O(m)
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map1 = new HashMap<>();
//        Map<Integer, String> result = new HashMap<>();
        List<String> result = new ArrayList<>();
        int index = Integer.MAX_VALUE;
        for (int i = 0; i < list1.length; i++) {
            map1.put(list1[i], i);
        }
        for (int i = 0; i < list2.length; i++) {
            if (map1.containsKey(list2[i])) {
                if (i + map1.get(list2[i]) < index) {
                    result.clear();
                    result.add(list2[i]);
                    index = i + map1.get(list2[i]);
                } else if (i + map1.get(list2[i]) == index) {
                    result.add(list2[i]);
                }

            }
        }
        return result.toArray(new String[0]);

    }

    //238. 除自身以外数组的乘积
    //给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
    //提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。
    //
    //说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
    //方法一：利用前缀乘积和后缀乘积
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    public int[] productExceptSelf(int[] nums) {
        int[] pres = new int[nums.length];
        int[] posts = new int[nums.length];
        pres[0] = 1;
        posts[nums.length - 1] = 1;
        int pretemp = 1;
        int postemp = 1;
        for (int i = 1; i < nums.length; i++) {
            pretemp *= nums[i - 1];
            pres[i] = pretemp;
            postemp *= nums[nums.length - i];
            posts[nums.length - i - 1] = postemp;
        }
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            result[i] = pres[i] * posts[i];
        }

        return result;
    }

    //方法二：空间复杂度 O(1) 的方法
    //思路
    //尽管上面的方法已经能够很好的解决这个问题，但是空间复杂度并不为常数。
    //由于输出数组不算在空间复杂度内，那么我们可以将 L 或 R 数组用输出数组来计算。先把输出数组当作 L 数组来计算，然后再动态构造 R 数组得到结果。让我们来看看基于这个思想的算法。
    //
    public int[] productExceptSelf2(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];

        // answer[i] 表示索引 i 左侧所有元素的乘积
        // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
        answer[0] = 1;
        for (int i = 1; i < length; i++) {
            answer[i] = nums[i - 1] * answer[i - 1];
        }

        // R 为右侧所有元素的乘积
        // 刚开始右边没有元素，所以 R = 1
        int R = 1;
        for (int i = length - 1; i >= 0; i--) {
            // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
            answer[i] = answer[i] * R;
            // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
            R *= nums[i];
        }
        return answer;
    }


    //1300. 转变数组后最接近目标值的数组和
    //给你一个整数数组 arr 和一个目标值 target ，请你返回一个整数 value ，使得将数组中所有大于 value 的值变成 value 后，数组的和最接近  target （最接近表示两者之差的绝对值最小）。
    //如果有多种使得和最接近 target 的方案，请你返回这些整数中的最小值。
    //请注意，答案不一定是 arr 中的数字。
    //示例 1：
    //输入：arr = [4,9,3], target = 10
    //输出：3
    //解释：当选择 value 为 3 时，数组会变成 [3, 3, 3]，和为 9 ，这是最接近 target 的方案。
    //示例 2：
    //输入：arr = [2,3,5], target = 10
    //输出：5
    //方法一：枚举 + 二分查找
    //枚举可能的value值，因为都是整数，所以可以确定value的下限是0，上限是数组的最大值
    //当枚举到 value = x 时，我们需要将数组 arr 中所有小于 x 的值保持不变，所有大于等于 x 的值变为 x。要实现这个操作，我们可以将数组 arr 先进行排序，随后进行二分查找，找出数组 arr 中最小的大于等于 x 的元素 arr[i]。此时数组的和变为
    //  arr[0] + ... + arr[i - 1] + x * (n - i)
    //使用该操作是因为很多编程语言自带的二分查找只能返回目标值第一次出现的位置。在此鼓励读者自己实现返回目标值最后一次出现的位置的二分查找。
    //
    //为了加速求和操作，我们可以预处理出数组 arr 的前缀和，这样数组求和的时间复杂度即能降为 O(1)。我们将和与 target 进行比较，同时更新答案即可。
    //时间复杂度：O((N+C)logN)，其中 N 是数组 arr 的长度，C 是一个常数，为数组 arr 中的最大值，不会超过 10^5，排序需要的时间复杂度为 O(NlogN)，二分查找的单次时间复杂度为 O(logN)，需要进行 C 次。
    //
    //空间复杂度：O(N)。我们需要 O(N) 的空间用来存储数组 arr 的前缀和，排序需要 O(logN) 的栈空间，因此最后总空间复杂度为 O(N)。
    //
    public int findBestValue(int[] arr, int target) {
        //先由小到大排序
        Arrays.sort(arr);
        int n = arr.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        int r = arr[n - 1];
        int ans = 0, diff = target;
        for (int i = 1; i <= r; ++i) {
            int index = Arrays.binarySearch(arr, i); //返回第一个大于i的位置
            if (index < 0) {
                index = -index - 1;
            }
            int cur = prefix[index] + (n - index) * i;
            if (Math.abs(cur - target) < diff) {
                ans = i;
                diff = Math.abs(cur - target);
            }
        }
        return ans;
    }

    //方法二：双重二分查找
    //方法一的枚举策略建立在数组 arr 的元素范围不大的条件之上。如果数组 arr 中的元素范围是 [1,10^9]，那么我们将无法直接枚举，有没有更好的解决方法呢？
    //我们首先考虑题目的一个简化版本：我们需要找到 value，使得数组的和最接近 target 且不大于 target。
    //可以发现，在 [0,max(arr)]（即方法一中确定的上下界）的范围之内，随着 value 的增大，数组的和是严格单调递增的。
    // 这里「严格」的意思是，不存在两个不同的 value 值，它们对应的数组的和相等。这样一来，一定存在唯一的一个 value 值，使得数组的和最接近且不大于 target。
    // 并且由于严格单调递增的性质，我们可以通过二分查找的方法，找到这个 value 值，记为 value_lower。
    //
    //同样地，我们考虑题目的另一个简化版本：我们需要找到一个 value，使得数组的和最接近 target 且大于 target。
    // 我们也可以通过二分查找的方法，找到这个 value 值，记为 value_upper。
    //
    //显然 value 值就是 value_lower 和 value_upper 中的一个，
    // 我们只需要比较这两个值对应的数组的和与 target 的差，就能确定最终的答案。
    // 这样一来，我们通过两次二分查找，就可以找出 value 值，在每一次二分查找中，我们使用和方法一中相同的查找方法，快速地求出每个 value 值对应的数组的和。
    // 算法从整体上来看，是外层二分查找中嵌套了一个内层二分查找。
    //
    //那么这个方法还有进一步优化的余地吗？
    // 仔细思考一下 value_lower 与 value_upper 的定义，前者最接近且不大于 target，后者最接近且大于 target。
    // 由于数组的和随着 value 的增大是严格单调递增的，所以 value_upper 的值一定就是 value_lower + 1。因此我们只需要进行一次外层二分查找得到 value_lower，并直接通过 value_lower + 1 计算出 value_upper 的值就行了。这样我们就减少了一次外层二分查找，虽然时间复杂度没有变化，但降低了常数。
    public int findBestValue2(int[] arr, int target) {
        Arrays.sort(arr);
        int n = arr.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        int l = 0, r = arr[n - 1], ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            int index = Arrays.binarySearch(arr, mid);
            if (index < 0) {
                index = -index - 1;
            }
            int cur = prefix[index] + (n - index) * mid;
            if (cur <= target) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        int chooseSmall = check(arr, ans);
        int chooseBig = check(arr, ans + 1);
        return Math.abs(chooseSmall - target) <= Math.abs(chooseBig - target) ? ans : ans + 1;
    }

    public int check(int[] arr, int x) {
        int ret = 0;
        for (int num : arr) {
            ret += Math.min(num, x);
        }
        return ret;
    }


    //11. 盛最多水的容器
    //你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
    // 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。
    // 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
    //
    //说明：你不能倾斜容器，且 n 的值至少为 2。
    //方法一：暴力法
    //时间复杂度：O(n^2)
    public int maxArea(int[] height) {
        int result = 0;
        for (int i = 0; i < height.length - 1; i++) {
            int maxTemp = 0;
            for (int j = i + 1; j < height.length; j++) {
                int less = height[i] < height[j] ? height[i] : height[j];
                if (maxTemp < less * (j - i)) {
                    maxTemp = less * (j - i);
                }
            }
            if (result < maxTemp) result = maxTemp;
        }
        return result;
    }

    //方法二：双指针
    //算法流程： 设置双指针 i,j 分别位于容器壁两端，根据规则移动指针（后续说明），并且更新面积最大值 res，直到 i == j 时返回 res。
    //
    //指针移动规则与证明： 每次选定围成水槽两板高度 h[i],h[j] 中的短板，向中间收窄 1 格。以下证明：
    //设每一状态下水槽面积为 S(i,j),(0<=i<j<n)，由于水槽的实际高度由两板中的短板决定，则可得面积公式S(i,j)=min(h[i],h[j])×(j−i)。
    //在每一个状态下，无论长板或短板收窄 1 格，都会导致水槽 底边宽度 −1：
    //若向内移动短板，水槽的短板 min(h[i],h[j]) 可能变大，因此水槽面积 S(i,j) 可能增大。
    //若向内移动长板，水槽的短板 min(h[i],h[j]) 不变或变小，下个水槽的面积一定小于当前水槽面积。
    //因此，向内收窄短板可能获取面积最大值
    //时间复杂度 O(N)，双指针遍历一次底边宽度 N 。
    //空间复杂度 O(1)，指针使用常数额外空间。
    public int maxArea2(int[] height) {
        int i = 0, j = height.length - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                    Math.max(res, (j - i) * height[i++]) :
                    Math.max(res, (j - i) * height[j--]);
        }
        return res;
    }

    //1014. 最佳观光组合
    //给定正整数数组 A，A[i] 表示第 i 个观光景点的评分，并且两个景点 i 和 j 之间的距离为 j - i。
    //一对景点（i < j）组成的观光组合的得分为（A[i] + A[j] + i - j）：景点的评分之和减去它们两者之间的距离。
    //返回一对观光景点能取得的最高分。
    //示例：
    //输入：[8,1,5,2,6]
    //输出：11
    //解释：i = 0, j = 2, A[i] + A[j] + i - j = 8 + 5 + 0 - 2 = 11
    //方法一：暴力法   提交报超时
    public int maxScoreSightseeingPair(int[] A) {
        int result = Integer.MIN_VALUE;
        for (int i = 0; i < A.length - 1; i++) {
            for (int j = i + 1; j < A.length; j++) {
                int now = A[i] + A[j] + i - j;
                if (result <= now) {
                    result = now;
                }
            }
        }
        return result;
    }

    //方法二：
    //我们考虑从前往后枚举 j 来统计答案，对于每个观光景点 j 而言，我们需要遍历 [0,j−1] 的观光景点 i 来计算组成观光组合 (i,j) 得分的最大值cntj
    //来作为第 j 个观光景点的值，那么最后的答案无疑就是所有观光景点值的最大值，即max{cntj}   j=0...n-1 .
    //但是枚举 j 需要 O(n)的时间复杂度，遍历[0,j−1] 的观光景点 i 也需要 O(n) 的时间复杂度，因此该方法总复杂度为 O(n^2)，
    // 不能通过所有测试用例，我们需要进一步优化时间复杂度。
    //
    //我们回过头来看得分公式，我们可以将其拆分成 A[i]+i 和 A[j]−j 两部分，这样对于统计景点 j 答案的时候，
    // 由于 A[j]−j 是固定不变的，因此最大化 A[i]+i+A[j]−j 的值其实就等价于求 [0,j−1] 中A[i]+i 的最大值 mx，
    // 景点 j 的答案即为 mx+A[j]−j 。而 mx 的值我们只要从前往后枚举 j 的时候同时维护即可，
    // 这样每次枚举景点 j 的时候，寻找使得得分最大的 i 就能从 O(n) 降至 O(1) 的时间复杂度，总时间复杂度就能从 O(n^2) 降至 O(n)。
    //
    //时间复杂度：O(n)，其中 n 为数组 A 的大小。我们只需要遍历一遍数组即可。
    //空间复杂度：O(1)。我们只需要常数空间来存放若干变量。
    public int maxScoreSightseeingPair2(int[] A) {
        int ans = 0, mx = A[0] + 0;
        for (int j = 1; j < A.length; ++j) {
            ans = Math.max(ans, mx + A[j] - j);
            // 边遍历边维护
            mx = Math.max(mx, A[j] + j);
        }
        return ans;
    }

    //448. 找到所有数组中消失的数字
    //给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
    //找到所有在 [1, n] 范围之间没有出现在数组中的数字。
    //您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
    //
    //示例:
    //输入:
    //[4,3,2,7,8,2,3,1]
    //输出:
    //[5,6]

    //方法一：我们假设数组大小为 N，它应该包含从 1 到 N 的数字。
    // 但是有些数字丢失了，我们要做的是记录我们在数组中遇到的数字。
    // 然后从 1....N 检查哈希表中没有出现的数字。
    //时间复杂度：O(N)。
    //空间复杂度：O(N)。
    public static List<Integer> findDisappearedNumbers(int[] nums) {
        // Hash table for keeping track of the numbers in the array
        // Note that we can also use a set here since we are not
        // really concerned with the frequency of numbers.
        HashMap<Integer, Boolean> hashTable = new HashMap<Integer, Boolean>();
        for (int i = 0; i < nums.length; i++) {
            hashTable.put(nums[i], true);
        }
        List<Integer> result = new LinkedList<Integer>();
        // Iterate over the numbers from 1 to N and add all those
        // that don't appear in the hash table.
        for (int i = 1; i <= nums.length; i++) {
            if (!hashTable.containsKey(i)) {
                result.add(i);
            }
        }
        return result;
    }

    //方法二：原地修改，标记
    //我们需要知道数组中存在的数字，由于数组的元素取值范围是 [1, N]，所以我们可以不使用额外的空间去解决它。
    //我们可以在输入数组本身以某种方式标记已访问过的数字，然后再找到缺失的数字。
    //算法：
    //
    //遍历输入数组的每个元素一次。
    //我们将把 |nums[i]|-1 索引位置的元素标记为负数。即 nums[∣nums[i]∣−1]×−1 。
    //然后遍历数组，若当前数组元素 nums[i] 为负数，说明我们在数组中存在数字 i+1。
    //时间复杂度：O(N)。
    //空间复杂度：O(1)，因为我们在原地修改数组，没有使用额外的空间。
    public List<Integer> findDisappearedNumbers2(int[] nums) {
        // Iterate over each of the elements in the original array
        for (int i = 0; i < nums.length; i++) {
            // Treat the value as the new index
            int newIndex = Math.abs(nums[i]) - 1;
            // Check the magnitude of value at this new index
            // If the magnitude is positive, make it negative
            // thus indicating that the number nums[i] has
            // appeared or has been visited.
            if (nums[newIndex] > 0) {
                nums[newIndex] *= -1;
            }
        }
        // Response array that would contain the missing numbers
        List<Integer> result = new LinkedList<Integer>();
        // Iterate over the numbers from 1 to N and add all those
        // that have positive magnitude in the array
        for (int i = 1; i <= nums.length; i++) {
            if (nums[i - 1] > 0) {
                result.add(i);
            }
        }
        return result;
    }


    //41. 缺失的第一个正数
    //给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
    //
    //示例 1:
    //输入: [1,2,0]
    //输出: 3
    //示例 2:
    //输入: [3,4,-1,1]
    //输出: 2
    //示例 3:
    //输入: [7,8,9,11,12]
    //输出: 1
    // 
    //提示：
    //你的算法的时间复杂度应为O(n)，并且只能使用常数级别的额外空间。
    //前言
    //如果本题没有额外的时空复杂度要求，那么就很容易实现：
    //  我们可以将数组所有的数放入哈希表，随后从 1 开始依次枚举正整数，并判断其是否在哈希表中；
    //  我们可以从 1 开始依次枚举正整数，并遍历数组，判断其是否在数组中。
    //
    //如果数组的长度为 N，那么第一种做法的时间复杂度为 O(N)，空间复杂度为 O(N)；
    // 第二种做法的时间复杂度为 O(N^2)，空间复杂度为 O(1)。但它们都不满足题目的要求：时间复杂度为 O(N)，空间复杂度为 O(1)。
    //「真正」满足此要求的算法是不存在的。但是我们可以退而求其次：
    // 利用给定数组中的空间来存储一些状态。也就是说，如果题目给定的数组是不可修改的，那么就不存在满足时空复杂度要求的算法；
    // 但如果我们可以修改给定的数组，那么是存在满足要求的算法的。
    //方法一：哈希表
    //对于「前言」中提到的第一种做法：
    //我们可以将数组所有的数放入哈希表，随后从 1 开始依次枚举正整数，并判断其是否在哈希表中。
    //
    //仔细想一想，我们为什么要使用哈希表？这是因为哈希表是一个可以支持快速查找的数据结构：
    //  给定一个元素，我们可以在 O(1) 的时间查找该元素是否在哈希表中。因此，我们可以考虑将给定的数组设计成哈希表的「替代产品」。
    //实际上，对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 [1, N+1] 中。这是因为如果[1,N] 都出现了，那么答案是 N+1，
    // 否则答案是 [1, N]中没有出现的最小正整数。这样一来，我们将所有在 [1,N] 范围内的数放入哈希表，也可以得到最终的答案。
    // 而给定的数组恰好长度为 N，这让我们有了一种将数组设计成哈希表的思路：
    //
    //我们对数组进行遍历，对于遍历到的数 x，如果它在 [1,N] 的范围内，那么就将数组中的第 x−1 个位置（注意：数组下标从 0 开始）打上「标记」。
    // 在遍历结束之后，如果所有的位置都被打上了标记，那么答案是N+1，否则答案是最小的没有打上标记的位置加 1。
    //
    //那么如何设计这个「标记」呢？由于数组中的数没有任何限制，因此这并不是一件容易的事情。
    // 但我们可以继续利用上面的提到的性质：由于我们只在意 [1,N] 中的数，
    // 因此我们可以先对数组进行遍历，把不在 [1,N] 范围内的数修改成任意一个大于 N 的数（例如 N+1）。
    // 这样一来，数组中的所有数就都是正数了，因此我们就可以将「标记」表示为「负号」。算法的流程如下：
    //
    //我们将数组中所有小于等于 0 的数修改为 N+1；
    //我们遍历数组中的每一个数 x，它可能已经被打了标记，因此原本对应的数为 |x|，其中 ∣∣ 为绝对值符号。
    //  如果 |x| ∈[1,N]，那么我们给数组中的第 ∣x∣−1 个位置的数添加一个负号。注意如果它已经有负号，不需要重复添加；
    //
    //在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。
    //时间复杂度：O(N)，其中 N 是数组的长度。
    //空间复杂度：O(1)。
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }

    //方法二：置换
    //除了打标记以外，我们还可以使用置换的方法，将给定的数组「恢复」成下面的形式：
    //如果数组中包含 x∈[1,N]，那么恢复后，数组的第 x−1 个元素为 x。
    //在恢复后，数组应当有 [1, 2, ..., N] 的形式，但其中有若干个位置上的数是错误的，每一个错误的位置就代表了一个缺失的正数。
    // 以题目中的示例二 [3, 4, -1, 1] 为例，恢复后的数组应当为 [1, -1, 3, 4]，我们就可以知道缺失的数为 2。
    //
    //那么我们如何将数组进行恢复呢？我们可以对数组进行一次遍历，
    // 对于遍历到的数 x=nums[i]，如果 x∈[1,N]，我们就知道 x 应当出现在数组中的x−1 的位置，因此交换 nums[i] 和nums[x−1]，
    // 这样 x 就出现在了正确的位置。在完成交换后，新的 nums[i] 可能还在 [1,N] 的范围内，我们需要继续进行交换操作，直到 x不在 [1, N]。
    //
    //注意到上面的方法可能会陷入死循环。如果 nums[i] 恰好与 nums[x−1] 相等，那么就会无限交换下去。
    // 此时我们有 nums[i]=x=nums[x−1]，说明 x 已经出现在了正确的位置。因此我们可以跳出循环，开始遍历下一个数。
    //
    //由于每次的交换操作都会使得某一个数交换到正确的位置，因此交换的次数最多为 N，整个方法的时间复杂度为 O(N)。
    //时间复杂度：O(N)，其中 N 是数组的长度。
    //空间复杂度：O(1)。
    public int firstMissingPositive2(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = temp;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    //442. 数组中重复的数据
    //给定一个整数数组 a，其中1 ≤ a[i] ≤ n （n为数组长度）, 其中有些元素出现两次而其他元素出现一次。
    //找到所有出现两次的元素。
    //你可以不用到任何额外空间并在O(n)时间复杂度内解决这个问题吗？
    //
    //示例：
    //输入:
    //[4,3,2,7,8,2,3,1]
    //输出:
    //[2,3]
    //解题思路
    //找到数字i时，将位置i-1处的数字翻转为负数。
    //如果位置i-1 上的数字已经为负，则i是出现两次的数字。
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < nums.length; ++i) {
            int index = Math.abs(nums[i]) - 1;
            if (nums[index] < 0)
                res.add(Math.abs(index + 1));
            nums[index] = -nums[index];
        }
        return res;
    }


    //718. 最长重复子数组
    //给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
    //
    //示例 1:
    //输入:
    //A: [1,2,3,2,1]
    //B: [3,2,1,4,7]
    //输出: 3
    //解释:
    //长度最长的公共子数组是 [3, 2, 1]。

    //暴力法
    //即枚举数组 A 中的起始位置 i 与数组 B 中的起始位置 j，然后计算 A[i:] 与 B[j:] 的最长公共前缀 k。最终答案即为所有的最长公共前缀的最大值。
    //虽然该暴力解法的最坏时间复杂度为 O(n^3)，但通过观察该暴力解法，我们可以提出几个时间复杂度更优秀的解法。
    //方法一：动态规划法
    //思路及算法
    //
    //暴力解法的过程中，我们发现最坏情况下对于任意 i 与 j ，A[i] 与 B[j] 比较了 min(i+1,j+1) 次。
    // 这也是导致了该暴力解法时间复杂度过高的根本原因。
    //不妨设 A 数组为 [1, 2, 3]，B 两数组为为 [1, 2, 4] ，那么在暴力解法中 A[2] 与 B[2] 被比较了三次。
    // 这三次比较分别是我们计算 A[0:] 与 B[0:] 最长公共前缀、 A[1:] 与 B[1:] 最长公共前缀以及 A[2:] 与 B[2:] 最长公共前缀时产生的。
    //
    //我们希望优化这一过程，使得任意一对 A[i] 和 B[j] 都只被比较一次。这样我们自然而然想到利用这一次的比较结果。
    // 如果 A[i] == B[j]，那么我们知道 A[i:] 与 B[j:] 的最长公共前缀为 A[i + 1:] 与 B[j + 1:] 的最长公共前缀的长度加一，否则我们知道 A[i:] 与 B[j:] 的最长公共前缀为零。
    //
    //这样我们就可以提出动态规划的解法：令 dp[i][j] 表示 A[i:] 和 B[j:] 的最长公共前缀，那么答案即为所有 dp[i][j] 中的最大值。
    // 如果 A[i] == B[j]，那么 dp[i][j] = dp[i + 1][j + 1] + 1，否则 dp[i][j] = 0。
    //
    //这里借用了 Python 表示数组的方法，A[i:] 表示数组 A 中索引 i 到数组末尾的范围对应的子数组。
    //考虑到这里 dp[i][j] 的值从 dp[i + 1][j + 1] 转移得到，所以我们需要倒过来，首先计算 dp[len(A) - 1][len(B) - 1]，最后计算 dp[0][0]。
    //
    //时间复杂度： O(N×M)
    //空间复杂度： O(N×M)
    //    N 表示数组 A 的长度，M 表示数组 B 的长度。
    //空间复杂度还可以再优化，利用滚动数组可以优化到 O(min(N,M))
    public int findLength(int[] A, int[] B) {
        int n = A.length, m = B.length;
        int[][] dp = new int[n + 1][m + 1];
        int ans = 0;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = m - 1; j >= 0; j--) {
                dp[i][j] = A[i] == B[j] ? dp[i + 1][j + 1] + 1 : 0;
                ans = Math.max(ans, dp[i][j]);
            }
        }
        return ans;
    }

    //方法二：滑动窗口
    //我们注意到之所以两位置会比较多次，是因为重复子数组在两个数组中的位置可能不同。以 A = [3, 6, 1, 2, 4], B = [7, 1, 2, 9] 为例，它们的最长重复子数组是 [1, 2]，在 A 与 B 中的开始位置不同。
    //但如果我们知道了开始位置，我们就可以根据它们将 A 和 B 进行「对齐」，即：
    //
    //A = [3, 6, 1, 2, 4]
    //B =    [7, 1, 2, 9]
    //           ↑ ↑
    //我们可以枚举 A 和 B 所有的对齐方式。对齐的方式有两类：第一类为 A 不变，B 的首元素与 A 中的某个元素对齐；第二类为 B 不变，A 的首元素与 B 中的某个元素对齐。
    // 对于每一种对齐方式，我们计算它们相对位置相同的重复子数组即可。
    //时间复杂度： O((N+M)×min(N,M))。
    //空间复杂度： O(1)。
    public int findLength2(int[] A, int[] B) {
        int n = A.length, m = B.length;
        int ret = 0;
        for (int i = 0; i < n; i++) {
            int len = Math.min(m, n - i);
            int maxlen = maxLength(A, B, i, 0, len);
            ret = Math.max(ret, maxlen);
        }
        for (int i = 0; i < m; i++) {
            int len = Math.min(n, m - i);
            int maxlen = maxLength(A, B, 0, i, len);
            ret = Math.max(ret, maxlen);
        }
        return ret;
    }

    public int maxLength(int[] A, int[] B, int addA, int addB, int len) {
        int ret = 0, k = 0;
        for (int i = 0; i < len; i++) {
            if (A[addA + i] == B[addB + i]) {
                k++;
            } else {
                k = 0;
            }
            ret = Math.max(ret, k);
        }
        return ret;
    }
    //方法三：二分查找 + 哈希
    //比较复杂，没看

    //209. 长度最小的子数组
    //给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。
    //如果不存在符合条件的连续子数组，返回 0。
    //示例：
    //输入：s = 7, nums = [2,3,1,2,4,3]
    //输出：2
    //解释：子数组 [4,3] 是该条件下的长度最小的连续子数组。
    // 
    //进阶：
    //如果你已经完成了 O(n) 时间复杂度的解法, 请尝试 O(nlogn) 时间复杂度的解法。
    //方法一：前缀和
    //时间复杂度：O(n^2)
    public int minSubArrayLen(int s, int[] nums) {
        int[] prefix = new int[nums.length];
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            prefix[i] = sum;
        }
        int result = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i; j < nums.length; j++) {
                if (prefix[j] - prefix[i] + nums[i] >= s) {
                    result = Math.min(result, j - i + 1);
                }
            }
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }
    //方法二：前缀和 + 二分查找
    //方法一的时间复杂度是 O(n^2)，因为在确定每个子数组的开始下标后，找到长度最小的子数组需要 O(n) 的时间。
    // 如果使用二分查找，则可以将时间优化到O(logn)。
    //
    //为了使用二分查找，需要额外创建一个数组 sums 用于存储数组 nums 的前缀和，其中 sums[i] 表示从 nums[0] 到 nums[i−1] 的元素和。
    // 得到前缀和之后，对于每个开始下标 i，可通过二分查找得到大于或等于 i 的最小下标 bound，使得sums[bound]−sums[i−1]≥s，并更新子数组的最小长度（此时子数组的长度是 bound−(i−1)）。
    //
    //因为这道题保证了数组中每个元素都为正，所以前缀和一定是递增的，这一点保证了二分的正确性。
    // 如果题目没有说明数组中每个元素都为正，这里就不能使用二分来查找这个位置了。

    //在很多语言中，都有现成的库和函数来为我们实现这里二分查找大于等于某个数的第一个位置的功能，
    // 比如 C++ 的 lower_bound，Java 中的 Arrays.binarySearch，C# 中的 Array.BinarySearch，Python 中的 bisect.bisect_left。
    // 但是有时面试官可能会让我们自己实现一个这样的二分查找函数
    //
    // 时间复杂度：O(nlogn)，其中 n 是数组的长度。需要遍历每个下标作为子数组的开始下标，遍历的时间复杂度是 O(n)，对于每个开始下标，需要通过二分查找得到长度最小的子数组，二分查找得时间复杂度是 O(logn)，因此总时间复杂度是O(nlogn)。
    //空间复杂度：O(n)，其中 n 是数组的长度。额外创建数组 sums 存储前缀和。
    //
    public int minSubArrayLen2(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int[] sums = new int[n + 1];
        // 为了方便计算，令 size = n + 1
        // sums[0] = 0 意味着前 0 个元素的前缀和为 0
        // sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
        // 以此类推
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= n; i++) {
            int target = s + sums[i - 1];
            //Arrays.binarySearch返回target的位置，如果没有找到返回应该插入的位置：(-(insertion point) - 1)
            int bound = Arrays.binarySearch(sums, target);
            if (bound < 0) {
                bound = -bound - 1;
            }
            if (bound <= n) {
                ans = Math.min(ans, bound - (i - 1));
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    //方法三：双指针
    //在方法一和方法二中，都是每次确定子数组的开始下标，然后得到长度最小的子数组，因此时间复杂度较高。为了降低时间复杂度，可以使用双指针的方法。
    //定义两个指针 start 和 end 分别表示子数组的开始位置和结束位置，维护变量 sum 存储子数组中的元素和（即从 nums[start] 到 nums[end] 的元素和）。
    //
    //初始状态下，start 和 end 都指向下标 0，sum 的值为 0。
    //每一轮迭代，将 nums[end] 加到 sum，如果 sum≥s，则更新子数组的最小长度
    // （此时子数组的长度是 end−start+1），然后将 nums[start] 从 sum 中减去并将 start 右移，直到 sum<s，在此过程中同样更新子数组的最小长度。
    // 在每一轮迭代的最后，将 end 右移。
    //时间复杂度：O(n)，其中 n 是数组的长度。指针 start 和 end 最多各移动 n 次。
    //空间复杂度：O(1)。
    public int minSubArrayLen3(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int start = 0, end = 0;
        int sum = 0;
        while (end < n) {
            sum += nums[end];
            while (sum >= s) {
                ans = Math.min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }


    public static void main(String[] args) {
//        System.out.println(firstBadVersion(2126753390));
        System.out.println(guessNumber(10));
    }
}
