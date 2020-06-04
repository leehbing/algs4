package com.leetcode;

import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 19/4/2020 3:06 PM
 * @Author: lihongbing
 */
public class TTTT {


    //1. 两数之和
    //解法一：暴力法
    //时间复杂度：O(n^2)
    //空间复杂度：O(1)
    public static int[] twoSum(int[] nums, int target) {
//        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
//                    count++;
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }

    //一遍扫描+二分查找
    //时间复杂度：O(nlgn)
    //空间复杂度：O(1)
    public static int[] twoSum_1(int[] nums, int target) {
        int j = -1;
        for (int i = 0; i < nums.length; i++) {
            j = Arrays.binarySearch(nums, target - nums[i]);
            if (j > i) return new int[]{i, j};
        }
        return null;
    }

    //方法二：两遍哈希表
    //通过以空间换取速度的方式，我们可以将查找时间从 O(n) 降低到 O(1)。
    // 哈希表正是为此目的而构建的，它支持以近似恒定的时间进行快速查找。
    // 我用“近似”来描述，是因为一旦出现冲突，查找用时可能会退化到 O(n)。
    // 但只要你仔细地挑选哈希函数，在哈希表中进行查找的用时应当被摊销为O(1)。
    //第一次迭代中，我们将每个元素的值和它的索引添加到表中。
    // 然后，在第二次迭代中，我们将检查每个元素所对应的目标元素（target - nums[i]）是否存在于表中。注意，该目标元素不能是 nums[i] 本身！
    //时间复杂度：O(n)，我们把包含有 n 个元素的列表遍历两次。由于哈希表将查找时间缩短到 O(1) ，所以时间复杂度为 O(n)。
    //空间复杂度：O(n)，所需的额外空间取决于哈希表中存储的元素数量，该表中存储了 n 个元素。
    public int[] twoSum2(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement) && map.get(complement) != i) {
                return new int[]{i, map.get(complement)};
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }

    //方法三：一遍哈希表
    //事实证明，我们可以一次完成。在进行迭代并将元素插入到表中的同时，我们还会回过头来检查表中是否已经存在当前元素所对应的目标元素。
    // 如果它存在，那我们已经找到了对应解，并立即将其返回。
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    public int[] twoSum3(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }


    //15. 三数之和         题目要求不能包括重复的三元组,这个要求很高。
    //看的别人的题解
    //首先对数组进行排序，排序后固定一个数 nums[i]，再使用左右指针指向 nums[i]后面的两端，数字分别为 nums[L] 和 nums[R]，计算三个数的和 sum 判断是否满足为0，满足则添加进结果集
    //如果 nums[i]大于 0，则三数之和必然无法等于 0，结束循环
    //如果 nums[i] == nums[i-1]，则说明该数字重复，会导致结果重复，所以应该跳过
    //当 sum == 0 时，nums[L] == nums[L+1] 则会导致结果重复，应该跳过，L++
    //当 sum == 0 时，nums[R] == nums[R-1] 则会导致结果重复，应该跳过，R--
    //时间复杂度：O(n^2)，n 为数组长度
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList();
        int len = nums.length;
        if (nums == null || len < 3) return ans;
        Arrays.sort(nums);  // 排序
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 去重
            int L = i + 1;
            int R = len - 1;
            while (L < R) {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[L], nums[R]));
                    while (L < R && nums[L] == nums[L + 1]) L++; // 去重
                    while (L < R && nums[R] == nums[R - 1]) R--; // 去重
                    L++;
                    R--;
                } else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }
        return ans;
    }


    //16. 最接近的三数之和
    //解法一：暴力法
    //时间复杂度：O(n^3)
    public static int threeSumClosest(int[] nums, int target) {
        int sum = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                for (int k = j + 1; k < nums.length; k++) {
                    if (Math.abs(nums[i] + nums[j] + nums[k] - target) < Math.abs(sum - target)) {
                        sum = nums[i] + nums[j] + nums[k];
                    }
                }
            }
        }
        return sum;
    }

    //解法二：
    //首先进行数组排序，时间复杂度 O(nlogn)
    //在数组 nums 中，进行遍历，每遍历一个值利用其下标i，形成一个固定值 nums[i]
    //再使用前指针指向 start = i + 1 处，后指针指向 end = nums.length - 1 处，也就是结尾处
    //根据 sum = nums[i] + nums[start] + nums[end] 的结果，判断 sum 与目标 target 的距离，如果更近则更新结果 ans
    //同时判断 sum 与 target 的大小关系，因为数组有序，如果 sum > target 则 end--，如果 sum < target 则 start++，如果 sum == target 则说明距离为 0 直接返回结果
    //整个遍历过程，固定值为 n 次，双指针为 n 次，时间复杂度为 O(n^2)
    //总时间复杂度：O(nlogn) + O(n^2) = O(n^2)
    public static int threeSumClosest2(int[] nums, int target) {
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1;
            int end = nums.length - 1;
            while (start < end) {
                sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(sum - target) < Math.abs(ans - target)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return sum;
                }

            }

        }
        return ans;

    }


    public String defangIPaddr(String address) {
        char[] res = new char[address.length() + 8];
        int j = 0;
        for (int i = 0; i < address.length(); i++) {
            if (address.charAt(i) == '.') {
                res[j++] = '[';
                res[j++] = address.charAt(i);
                res[j++] = ']';
            } else {
                res[j++] = address.charAt(i);
            }
        }
        return new String(res);
    }


//    public static void main(String[] args) throws Exception{


//1534236469反转后肯定会溢出。

//    }


    //204. 计数质数 统计所有小于非负整数 n 的质数的数量。
    //一般思维，判断一个数x是否是质数，遍历2～sqrt(x)，判断能否被x整除，这样判断一个数字x的时间复杂度就是O(x^0.5)，找出所有的复杂度就是O(n^1.5)
    //厄拉多塞筛法O(N * logN)
    public static int countPrimes(int n) {
        boolean[] isPrim = new boolean[n];
        Arrays.fill(isPrim, true); //初始化默认值都为 true
        for (int i = 2; i * i < n; i++) //只需要遍历到从2到sqrt(n)
            if (isPrim[i])
                for (int j = i * i; j < n; j += i) //将2*i改成了i*i更加高效
                    isPrim[j] = false;

        int count = 0;
        for (int i = 2; i < n; i++)
            if (isPrim[i]) count++;

        return count;


    }

    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        String temp = strs[0];
        String result = temp;
        for (int i = 1; i < strs.length; i++) {
            result = "";
            for (int j = 0; j < Math.min(temp.length(), strs[i].length()); j++) {
                if (temp.charAt(j) == strs[i].charAt(j)) result += temp.charAt(j);
                else break;
            }
            if (result.isEmpty()) break;
            temp = result;

        }

        return result;

    }

    public static boolean isValid(String s) {
        if (s.isEmpty()) return true;
        Stack stack = new Stack<Character>();
        for (int i = 0; i < s.length(); i++) {
            if (stack.isEmpty()) {
                stack.push(s.charAt(i));
                continue;
            }
            char peek = (Character) stack.peek();
            char cur = s.charAt(i);
            if ((peek == '(' && cur == ')') || (peek == '{' && cur == '}') || (peek == '[' && cur == ']')) {
                stack.pop();
            } else {
                stack.push(s.charAt(i));
            }
        }

        return stack.isEmpty();
    }


    public static int strStr(String haystack, String needle) {
        if (haystack.isEmpty() && needle.isEmpty()) return 0;
        if (needle.isEmpty()) return 0;

        int j = 0;
        for (int i = 0; i < haystack.length() - needle.length() + 1; i++) {

            for (j = 0; j < needle.length(); j++) {
                if (haystack.charAt(i + j) != needle.charAt(j)) {
                    break;
                }
            }
            if (j == needle.length()) return i;

        }

        return -1;
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

    //347. 前 K 个高频元素
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


    //448. 找到所有数组中消失的数字
    public static List<Integer> findDisappearedNumbers(int[] nums) {
        //方法一，我们假设数组大小为 N，它应该包含从 1 到 N 的数字。
        // 但是有些数字丢失了，我们要做的是记录我们在数组中遇到的数字。
        // 然后从 1....N 检查哈希表中没有出现的数字。
        //时间复杂度：O(N)。
        //空间复杂度：O(N)。
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

    //27. 移除元素    仅使用 O(1) 额外空间
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

    //152. 乘积最大子数组
    public static int maxProduct(int[] nums) {
        //方法一，暴力法
        //时间复杂度：O(n^2 + n)
        //空间复杂度：O(n)
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


    public static void main(String[] args) {
//        System.out.println(convert("A", 1));
//        System.out.println(romanToInt("MCMXCIV"));
//        String[] param = {"flower", "flow", "flight"};
//        System.out.println(longestCommonPrefix(param));
//        System.out.println(isValid("([)"));
//        System.out.println(strStr("aa", "aa"));
//        findAnagrams("cbaebabacd", "abc");

//        int k = 3;
//        int[] arr = {4, 5, 8, 2};
//        KthLargest kthLargest = new KthLargest(3, arr);
//        kthLargest.add(3);// returns 4
//        kthLargest.add(5);// returns 5
//        kthLargest.add(10);// returns 5
//        kthLargest.add(9);// returns 8
//        kthLargest.add(4);// returns 8
//        reverseBetween(Utils.stringToListNode("[1,2,3,4,5]"), 2, 4);
//        validPalindrome("abc");
//        isPalindrome("A man, a plan, a canal: Panama");
//        System.out.println(Math.abs(Integer.MAX_VALUE - 0));
        int[] param = {3, 7, 0, 2, 7, 0, 0, 7, 9, 8, 2, 6, 4, 3, 6, 9, 0, 9, 6, 7, 5, 9, 5, 2, 6, 5, 5, 8, 9, 2, 0, 1, 8, 7, 0, 3, 8, 5, 6, 0, 7, 2, 6, 9, 6, 5, 4, 4, 2, 8, 5, 7, 9, 0, 1, 8, 6, 1, 2, 7, 8, 1, 0, 3, 4, 8, 0, 0, 5, 5, 9, 6, 3, 4, 5, 6, 2, 9, 7, 7, 1, 1, 3, 8, 0, 6, 2, 5, 3, 3, 7, 2, 8, 9, 0, 1, 9, 5, 1, 1, 9, 7, 3, 2, 2, 1, 2, 1, 7, 4, 8, 9, 2, 7, 2, 4, 7, 8, 3, 4, 9, 8, 9, 1, 4, 8, 3, 6, 6, 7, 8, 0, 3, 5, 2, 3, 2, 2, 0, 5, 5, 7, 6, 3, 6, 8, 3, 2, 5, 0, 8, 2, 7, 4, 4, 3, 8, 2, 2, 4, 9, 3, 4, 2, 2, 9, 2, 4, 6, 1, 6, 2, 5, 0, 7, 4, 7, 4, 6, 3, 2, 7, 1, 9, 7, 0, 5, 2, 2, 0, 7, 7, 4, 0, 1, 3, 5, 2, 8, 7, 1, 5, 2, 1, 9, 4, 2, 8, 0, 4, 2, 2, 0, 2, 0, 1, 4, 0, 8, 1, 2, 5, 1, 2, 3, 4, 5, 5, 6, 1, 6, 4, 3, 2, 8, 6, 0, 1, 5, 7, 0, 0, 3, 2, 0, 8, 9, 4, 4, 8, 2, 3, 4, 9, 0, 3, 9, 3, 4, 3, 2, 0, 8, 2, 3, 6, 5, 7, 1, 2, 9, 4, 6, 9, 4, 5, 1, 2, 7, 2, 8, 4, 6, 2, 6, 8, 0, 2, 8, 4, 9, 6, 7, 4, 6, 3, 7, 6, 5, 1, 0, 2, 0, 1, 8, 5, 6, 9, 5, 9, 8, 9, 0, 1, 6, 5, 4, 4, 2, 5, 4, 6, 6, 0, 2, 3, 5, 5, 4, 2, 5, 6, 7, 3, 3, 7, 8, 3, 4, 1, 9, 6, 1, 2, 2, 5, 2, 6, 7, 9, 3, 5, 0, 4, 0, 9, 7, 7, 7, 4, 5, 0, 5, 9, 6, 2, 1, 0, 0, 1, 8, 7, 3, 0, 7, 5, 6, 5, 0, 9, 8, 9, 2, 1, 9, 3, 6, 9, 2, 8, 7, 7, 7, 3, 6, 3, 5, 4, 1, 4, 5, 0, 2, 1, 2, 8, 8, 1, 8, 5, 0, 2, 7, 6, 6, 4, 0, 9, 8, 1, 5, 4, 5, 5, 4, 3, 0, 9, 3, 9, 3, 4, 6, 4, 4, 0, 8, 8, 1, 8, 6, 5, 0, 1, 2, 8, 3, 6, 5, 6, 6, 3, 0, 1, 3, 1, 3, 2, 8, 5, 1, 4, 1, 7, 6, 2, 6, 9, 1, 3, 0, 1, 7, 5, 1, 4, 9, 7, 8, 6, 4, 5, 4, 1, 3, 4, 6, 7, 9, 8, 8, 3, 1, 8, 2, 8, 8, 8, 5, 8, 9, 4, 9, 5, 0, 5, 5, 4, 8, 2, 5, 9, 1, 6, 0, 2, 3, 0, 7, 2, 7, 9, 8, 2, 5, 3, 9, 8, 4, 6, 7, 0, 3, 5, 5, 9, 8, 9, 3, 7, 4, 4, 1, 2, 4, 0, 1, 5, 9, 6, 7, 7, 9, 8, 9, 1, 3, 9, 6, 7, 6, 7, 0, 6, 3, 5, 2, 5, 0, 6, 8, 1, 0, 1, 1, 4, 5, 0, 4, 7, 7, 0, 0, 8, 9, 5, 2, 4, 5, 0, 1, 5, 4, 2, 3, 1, 1, 8, 4, 8, 1, 9, 9, 7, 4, 3, 4, 1, 1, 1, 6, 0, 8, 6, 1, 5, 0, 7, 4, 7, 3, 4, 0, 6, 4, 3, 9, 4, 6, 4, 6, 8, 8, 2, 2, 7, 5, 3, 3, 3, 0, 1, 3, 7, 8, 4, 9, 0, 5, 3, 0, 8, 2, 2, 0, 8, 2, 2, 2, 2, 0, 1, 6, 6, 4, 9, 8, 0, 1, 7, 7, 5, 4, 4, 3, 1, 2, 3, 0, 3, 9, 2, 5, 4, 5, 9, 1, 6, 3, 3, 9, 0, 8, 4, 4, 4, 6, 6, 4, 7, 7, 0, 9, 5, 6, 4, 0, 2, 1, 8, 1, 9, 8, 6, 2, 4, 8, 0, 9, 2, 2, 4, 1, 4, 4, 8, 3, 8, 0, 2, 8, 4, 9, 2, 9, 7, 1, 3, 8, 6, 8, 0, 7, 9, 2, 6, 5, 5, 2, 6, 5, 5, 2, 4, 2, 9, 4, 1, 4, 6, 1, 7, 8, 3, 8, 5, 3, 9, 1, 2, 0, 3, 1, 2, 1, 7, 5, 4, 9, 1, 2, 6, 3, 4, 3, 6, 5, 7, 8, 8, 8, 9, 6, 5, 1, 7, 6, 0, 3, 9, 8, 2, 4, 6, 8, 0, 2, 7, 8, 4, 2, 8, 0, 6, 5, 4, 3, 0, 6, 3, 8, 8, 7, 5, 5, 0, 4, 3, 0, 2, 7, 5, 4, 8, 0, 6, 7, 9, 7, 7, 9, 0, 7, 2, 7, 2, 2, 5, 2, 0, 6, 5, 6, 2, 6, 2, 2, 1, 3, 2, 7, 7, 8, 2, 9, 3, 7, 3, 1, 8, 9, 6, 6, 9, 3, 0, 7, 5, 0, 7, 3, 1, 3, 6, 9, 7, 3, 2, 7, 0, 5, 2, 7, 4, 1, 1, 5, 0, 1, 4, 4, 7, 7, 2, 9, 8, 3, 1, 8, 5, 4, 5, 9, 3, 7, 7, 6, 7, 0, 0, 7, 3, 0, 7, 4, 8, 5, 0, 2, 1, 9, 3, 6, 3, 5, 3, 8, 7, 2, 2, 5, 3, 1, 3, 9, 8, 6, 2, 8, 7, 5, 7, 7, 1, 9, 3, 3, 5, 5, 6, 9, 5, 2, 7, 8, 3, 5, 0, 4, 7, 8, 4, 9, 7, 1, 1, 9, 5, 9, 4, 1, 1, 5, 8, 2, 2, 9, 0, 7, 8, 3, 1, 5, 1, 4, 7, 1, 8, 6, 3, 5, 2, 6, 2, 6, 4, 3, 1, 6, 9, 4, 9, 3, 9, 9, 8, 2, 8, 0, 8, 3, 2, 4, 0, 5, 9, 2, 4, 3, 7, 9, 0, 0, 5, 5, 7, 7, 6, 3, 9, 3, 8, 5, 4, 9, 1, 7, 1, 7, 1, 5, 7, 1, 4, 3, 4, 7, 3, 4, 2, 3, 7, 2, 0, 2, 4, 8, 7, 7, 1, 1, 8, 3, 0, 1, 4, 5, 3, 4, 1, 3, 2, 5, 3, 8, 1, 6, 0, 9, 7, 6, 3, 0, 3, 9, 3, 0, 2, 4, 0, 9, 0, 8, 7, 8, 7, 6, 9, 4, 0, 0, 4, 5, 4, 7, 8, 6, 6, 8, 0, 1, 9, 4, 9, 1, 0, 4, 8, 6, 1, 6, 6, 9, 6, 8, 4, 7, 6, 8, 8, 1, 0, 0, 9, 0, 8, 0, 5, 2, 3, 7, 7, 5, 7, 6, 3, 8, 2, 2, 3, 9, 1, 2, 9, 3, 1, 0, 5, 5, 2, 1, 5, 9, 4, 4, 0, 7, 9, 7, 3, 3, 4, 4, 3, 7, 7, 4, 6, 6, 3, 9, 6, 4, 6, 9, 8, 5, 0, 0, 2, 2, 4, 5, 6, 2, 9, 9, 1, 7, 0, 7, 6, 7, 0, 4, 9, 5, 8, 7, 3, 8, 6, 2, 5, 5, 0, 7, 8, 6, 8, 6, 3, 5, 9, 7, 9, 7, 5, 1, 0, 3, 8, 9, 0, 4, 7, 1, 0, 3, 2, 0, 3, 3, 2, 2, 6, 6, 7, 2, 1, 3, 5, 7, 9, 5, 9, 9, 0, 3, 5, 4, 4, 9, 2, 8, 3, 8, 3, 7, 6, 1, 3, 9, 7, 2, 4, 8, 0, 6, 0, 7, 7, 3, 5, 5, 0, 7, 4, 3, 4, 7, 8, 3, 0, 7, 7, 0, 0, 8, 9, 2, 7, 3, 8, 8, 2, 4, 3, 3, 3, 8, 1, 5, 2, 8, 0, 2, 0, 2, 6, 5, 2, 3, 6, 5, 8, 1, 3, 6, 9, 0, 4, 6, 1, 0, 6, 3, 8, 0, 2, 0, 2, 4, 9, 9, 7, 9, 3, 4, 7, 3, 4, 5, 2, 7, 1, 8, 8, 0, 2, 5, 0, 7, 7, 2, 1, 3, 7, 1, 2, 1, 1, 2, 5, 7, 1, 7, 1, 6, 3, 7, 9, 1, 4, 6, 4, 7, 5, 8, 0, 4, 8, 8, 5, 8, 6, 5, 1, 7, 7, 2, 4, 8, 3, 6, 1, 9, 3, 1, 5, 6, 6, 0, 7, 6, 1, 3, 6, 7, 9, 9, 9, 7, 2, 4, 2, 5, 3, 2, 2, 6, 3, 8, 4, 8, 9, 8, 5, 6, 3, 2, 9, 0, 2, 4, 0, 8, 4, 1, 9, 0, 6, 9, 5, 0, 3, 1, 0, 3, 2, 8, 2, 8, 9, 1, 6, 0, 3, 0, 2, 5, 0, 2, 4, 4, 2, 5, 5, 8, 6, 1, 1, 7, 8, 2, 9, 6, 5, 0, 2, 2, 2, 4, 0, 7, 5, 6, 0, 1, 8, 9, 3, 7, 5, 7, 8, 4, 8, 9, 5, 4, 8, 8, 5, 8, 1, 5, 8, 1, 7, 7, 6, 9, 6, 0, 0, 7, 1, 8, 1, 6, 9, 7, 0, 9, 1, 6, 8, 0, 4, 4, 4, 0, 5, 4, 2, 3, 7, 2, 8, 8, 1, 0, 8, 4, 7, 1, 5, 5, 6, 4, 2, 7, 3, 3, 5, 6, 0, 6, 0, 3, 6, 6, 5, 1, 8, 4, 2, 7, 1, 5, 3, 4, 7, 4, 2, 1, 2, 8, 7, 3, 4, 0, 1, 2, 2, 6, 0, 5, 5, 8, 6, 0, 8, 9, 6, 8, 6, 2, 8, 0, 7, 6, 0, 9, 4, 9, 7, 2, 9, 3, 0, 8, 9, 6, 9, 1, 1, 2, 2, 7, 4, 4, 8, 8, 7, 5, 1, 3, 5, 8, 0, 9, 9, 6, 6, 7, 3, 9, 3, 2, 4, 8, 4, 1, 6, 8, 5, 6, 3, 2, 5, 7, 6, 3, 6, 9, 8, 6, 1, 8, 6, 0, 1, 9, 4, 4, 0, 8, 1, 9, 0, 9, 6, 5, 9, 1, 6, 4, 4, 0, 1, 1, 0, 0, 3, 2, 4, 5, 4, 4, 5, 1, 7, 6, 4, 6, 8, 0, 8, 1, 8, 8, 0, 5, 6, 3, 9, 9, 4, 0, 4, 2, 8, 3, 7, 5, 0, 4, 4, 0, 4, 7, 9, 2, 2, 2, 6, 5, 0, 6, 8, 6, 1, 1, 7, 2, 5, 6, 1, 0, 6, 4, 4, 8, 4, 4, 1, 9, 2, 2, 3, 1, 5, 6, 9, 0, 9, 0, 6, 1, 5, 5, 2, 5, 7, 0, 9, 5, 7, 4, 4, 1, 4, 9, 5, 6, 3, 5, 7, 5, 1, 8, 4, 8, 3, 2, 3, 3, 0, 4, 1, 1, 2, 7, 1, 6, 2, 7, 7, 1, 7, 3, 4, 0, 3, 3, 4, 1, 5, 8, 5, 0, 5, 7, 5, 0, 9, 2, 3, 6, 4, 7, 5, 6, 1, 2, 3, 9, 3, 7, 4, 8, 4, 1, 4, 3, 6, 4, 0, 5, 5, 4, 3, 3, 8, 8, 7, 4, 0, 1, 3, 7, 4, 2, 7, 8, 1, 3, 0, 5, 2, 5, 8, 7, 5, 9, 8, 0, 5, 2, 9, 9, 2, 9, 2, 5, 9, 0, 2, 4, 1, 7, 8, 9, 4, 1, 6, 4, 9, 7, 4, 3, 3, 4, 6, 2, 7, 5, 9, 7, 6, 3, 3, 7, 7, 2, 1, 8, 5, 0, 6, 7, 0, 4, 7, 5, 2, 1, 1, 0, 2, 6, 3, 4, 0, 7, 8, 8, 6, 2, 3, 7, 3, 9, 7, 3, 2, 8, 0, 4, 2, 6, 6, 2, 7, 8, 0, 1, 6, 3, 0, 2, 4, 7, 3, 5, 5, 3, 9, 5, 7, 4, 6, 9, 4, 3, 4, 6, 4, 6, 5, 5, 0, 2, 9, 3, 2, 6, 3, 9, 2, 0, 4, 1, 3, 5, 3, 7, 4, 9, 8, 6, 6, 5, 0, 8, 9, 9, 9, 7, 4, 6, 3, 2, 6, 9, 8, 9, 2, 1, 3, 9, 1, 1, 4, 7, 2, 5, 2, 4, 2, 0, 1, 4, 9, 6, 1, 1, 6, 7, 0, 2, 7, 3, 8, 0, 3, 0, 6, 5, 9, 1, 3, 6, 2, 5, 9, 9, 8, 3, 3, 1, 5, 8, 3, 8, 0, 4, 9, 5, 5, 6, 4, 2, 7, 0, 9, 9, 6, 1, 1, 0, 5, 9, 4, 9, 4, 4, 2, 3, 5, 1, 8, 9, 6, 8, 8, 2, 6, 9, 4, 0, 8, 9, 9, 4, 7, 2, 3, 8, 7, 7, 8, 0, 1, 9, 8, 2, 0, 3, 0, 3, 8, 7, 7, 6, 6, 2, 8, 5, 6, 7, 9, 7, 8, 2, 8, 7, 3, 5, 1, 0, 7, 1, 0, 3, 5, 2, 4, 5, 1, 2, 6, 6, 4, 5, 5, 0, 6, 6, 0, 9, 9, 4, 1, 8, 8, 6, 5, 2, 6, 8, 3, 8, 9, 8, 1, 3, 7, 0, 3, 7, 4, 6, 7, 4, 1, 5, 3, 6, 1, 4, 9, 8, 9, 8, 1, 0, 2, 3, 0, 1, 1, 9, 0, 9, 2, 3, 0, 7, 7, 0, 8, 8, 5, 7, 7, 5, 4, 2, 4, 2, 2, 3, 3, 3, 9, 0, 2, 4, 6, 3, 6, 9, 4, 0, 2, 3, 2, 0, 1, 6, 0, 1, 0, 3, 5, 1, 0, 2, 3, 5, 5, 7, 9, 3, 9, 9, 5, 9, 7, 7, 4, 2, 1, 3, 8, 1, 4, 6, 2, 0, 7, 8, 7, 8, 0, 1, 5, 8, 3, 9, 6, 1, 8, 2, 7, 9, 7, 3, 8, 5, 6, 2, 1, 5, 8, 7, 9, 3, 2, 6, 0, 2, 8, 5, 8, 4, 7, 4, 7, 2, 4, 1, 6, 7, 4, 5, 8, 6, 2, 0, 4, 7, 9, 8, 9, 5, 6, 5, 4, 7, 9, 8, 8, 2, 5, 6, 5, 1, 7, 5, 6, 4, 3, 5, 7, 8, 3, 7, 9, 8, 1, 8, 7, 1, 1, 3, 2, 4, 4, 0, 0, 0, 9, 8, 0, 1, 0, 3, 1, 5, 4, 5, 2, 5, 6, 7, 6, 5, 4, 1, 8, 6, 8, 6, 0, 7, 8, 5, 5, 1, 3, 1, 4, 5, 0, 9, 0, 7, 3, 1, 7, 9, 3, 8, 2, 3, 3, 7, 6, 1, 4, 4, 2, 1, 3, 1, 7, 1, 0, 4, 3, 8, 0, 4, 5, 3, 3, 4, 6, 0, 0, 8, 7, 3, 1, 8, 9, 2, 6, 4, 0, 9, 8, 7, 0, 1, 2, 7, 8, 3, 9, 1, 0, 9, 5, 5, 0, 4, 7, 0, 8, 6, 0, 2, 3, 3, 8, 5, 2, 7, 5, 6, 5, 1, 2, 2, 3, 0, 1, 6, 7, 5, 0, 0, 8, 5, 1, 0, 7, 4, 0, 4, 8, 7, 8, 2, 4, 7, 0, 8, 8, 0, 2, 0, 6, 2, 4, 4, 9, 7, 5, 4, 2, 4, 1, 0, 3, 2, 1, 5, 7, 8, 7, 5, 4, 2, 6, 1, 8, 7, 9, 0, 1, 0, 1, 8, 7, 6, 7, 2, 1, 5, 6, 0, 3, 3, 0, 7, 0, 6, 3, 3, 5, 0, 3, 2, 4, 1, 7, 8, 5, 2, 0, 8, 5, 4, 7, 4, 5, 8, 0, 7, 6, 3, 6, 1, 3, 9, 1, 0, 3, 0, 7, 6, 0, 9, 2, 1, 8, 7, 2, 3, 2, 6, 3, 9, 7, 9, 2, 2, 9, 4, 0, 2, 3, 4, 1, 4, 3, 4, 3, 7, 6, 4, 4, 7, 6, 1, 1, 2, 7, 6, 0, 9, 9, 0, 0, 6, 8, 7, 3, 6, 1, 6, 3, 3, 2, 8, 2, 3, 4, 3, 1, 5, 4, 7, 5, 5, 1, 9, 3, 6, 9, 6, 6, 6, 2, 4, 4, 2, 2, 3, 9, 6, 7, 2, 1, 7, 6, 5, 7, 5, 6, 9, 3, 6, 6, 7, 4, 5, 5, 2, 9, 2, 4, 4, 2, 7, 5, 9, 7, 5, 7, 8, 6, 6, 0, 0, 2, 3, 9, 9, 0, 8, 5, 2, 9, 5, 2, 7, 9, 3, 9, 5, 0, 9, 8, 8, 7, 5, 5, 2, 3, 7, 0, 1, 6, 7, 4, 5, 1, 9, 5, 2, 1, 8, 4, 4, 1, 7, 8, 8, 1, 6, 9, 1, 8, 6, 0, 9, 8, 2, 2, 4, 3, 9, 6, 0, 3, 7, 6, 6, 9, 4, 7, 3, 3, 2, 4, 1, 7, 4, 7, 0, 4, 6, 9, 6, 8, 6, 9, 0, 3, 7, 6, 5, 3, 0, 4, 5, 4, 6, 0, 7, 0, 6, 5, 7, 9, 9, 1, 1, 3, 9, 6, 5, 6, 3, 2, 2, 1, 1, 6, 8, 5, 9, 5, 5, 6, 1, 7, 8, 9, 7, 7, 8, 8, 9, 9, 9, 5, 7, 0, 1, 2, 0, 7, 1, 9, 7, 3, 2, 8, 9, 2, 0, 7, 1, 5, 2, 4, 5, 1, 8, 0, 5, 6, 3, 6, 8, 5, 3, 4, 4, 3, 5, 8, 5, 6, 3, 7, 4, 1, 2, 8, 0, 6, 5, 1, 2, 6, 9, 6, 6, 2, 2, 6, 2, 4, 3, 9, 0, 2, 2, 8, 5, 0, 0, 8, 7, 8, 2, 7, 5, 7, 1, 1, 3, 9, 7, 5, 5, 5, 9, 0, 6, 9, 0, 5, 5, 0, 7, 6, 1, 2, 2, 8, 0, 5, 5, 9, 7, 5, 5, 3, 3, 3, 8, 5, 2, 7, 8, 9, 6, 1, 6, 4, 7, 9, 0, 0, 2, 2, 8, 2, 9, 3, 2, 1, 8, 4, 3, 4, 5, 9, 6, 9, 0, 4, 3, 7, 5, 4, 8, 4, 8, 3, 8, 3, 6, 7, 8, 6, 9, 3, 8, 6, 2, 1, 8, 8, 6, 9, 3, 2, 6, 2, 5, 2, 2, 1, 5, 8, 4, 1, 6, 9, 8, 3, 4, 5, 9, 3, 6, 5, 7, 7, 4, 7, 2, 3, 3, 5, 7, 7, 5, 7, 4, 6, 6, 0, 0, 1, 8, 6, 4, 8, 4, 8, 3, 2, 9, 2, 7, 3, 9, 0, 5, 2, 6, 5, 9, 9, 4, 3, 7, 7, 6, 3, 0, 7, 3, 4, 7, 2, 9, 2, 7, 1, 5, 1, 9, 0, 3, 5, 7, 0, 5, 6, 4, 6, 7, 0, 5, 4, 1, 9, 5, 4, 9, 0, 0, 4, 0, 2, 1, 8, 5, 2, 9, 8, 4, 7, 2, 0, 7, 3, 4, 7, 1, 2, 8, 1, 2, 8, 6, 9, 6, 0, 8, 0, 6, 5, 8, 2, 0, 6, 3, 9, 5, 9, 0, 9, 7, 9, 4, 3, 3, 1, 3, 7, 9, 8, 5, 5, 9, 1, 0, 4, 9, 5, 6, 6, 5, 2, 3, 4, 2, 1, 3, 2, 0, 9, 1, 4, 1, 6, 6, 8, 2, 4, 8, 7, 0, 2, 6, 9, 4, 2, 2, 5, 0, 3, 8, 4, 4, 2, 8, 9, 8, 8, 3, 3, 0, 7, 0, 1, 6, 6, 6, 9, 5, 4, 0, 9, 3, 5, 7, 8, 1, 9, 0, 2, 4, 4, 1, 0, 5, 8, 5, 6, 1, 1, 6, 9, 4, 4, 8, 1, 5, 2, 3, 0, 8, 5, 1, 2, 0, 5, 3, 9, 9, 4, 4, 4, 0, 8, 6, 6, 0, 3, 3, 1, 7, 8, 9, 1, 4, 4, 5, 8, 2, 8, 7, 6, 8, 5, 1, 0, 7, 5, 5, 3, 5, 9, 9, 2, 8, 5, 8, 7, 6, 7, 6, 6, 4, 8, 8, 8, 8, 0, 4, 8, 3, 0, 6, 1, 5, 9, 9, 6, 6, 1, 0, 7, 4, 1, 6, 1, 7, 4, 3, 4, 8, 4, 5, 3, 6, 9, 3, 2, 4, 4, 5, 9, 8, 2, 1, 2, 4, 6, 3, 9, 8, 6, 6, 3, 6, 7, 9, 6, 2, 6, 7, 9, 9, 3, 3, 1, 8, 5, 9, 6, 5, 7, 4, 0, 3, 0, 0, 5, 0, 0, 2, 5, 6, 1, 6, 4, 6, 1, 0, 7, 4, 8, 7, 7, 9, 3, 3, 0, 5, 8, 0, 4, 3, 3, 1, 8, 3, 2, 6, 5, 6, 6, 1, 3, 7, 9, 5, 7, 7, 4, 8, 1, 6, 7, 8, 4, 7, 2, 8, 8, 4, 7, 1, 5, 4, 1, 1, 8, 8, 1, 9, 9, 1, 0, 7, 0, 0, 4, 1, 0, 6, 7, 0, 8, 3, 3, 6, 0, 7, 3, 6, 3, 2, 0, 7, 6, 5, 1, 4, 5, 2, 7, 9, 3, 1, 6, 7, 4, 9, 3, 6, 5, 2, 8, 9, 1, 0, 0, 1, 7, 0, 6, 2, 6, 5, 7, 5, 0, 7, 5, 7, 9, 0, 0, 7, 7, 3, 2, 5, 1, 5, 9, 2, 2, 0, 7, 8, 0, 2, 6, 2, 9, 6, 9, 4, 4, 6, 1, 7, 3, 2, 0, 5, 0, 2, 2, 2, 6, 8, 6, 7, 8, 4, 7, 6, 6, 2, 5, 3, 2, 3, 6, 5, 6, 4, 2, 9, 0, 0, 9, 1, 2, 9, 9, 0, 1, 5, 3, 2, 3, 5, 0, 8, 1, 1, 6, 4, 8, 1, 9, 3, 3, 3, 8, 8, 0, 1, 5, 7, 1, 4, 6, 9, 4, 6, 3, 9, 6, 4, 7, 1, 7, 2, 0, 7, 9, 8, 3, 5, 2, 1, 2, 2, 6, 5, 5, 7, 6, 2, 9, 8, 2, 7, 1, 6, 4, 6, 9, 2, 9, 0, 4, 8, 4, 9, 1, 6, 9, 2, 0, 5, 2, 6, 0, 7, 3, 5, 4, 0, 0, 4, 2, 4, 5, 0, 5, 5, 8, 4, 6, 4, 4, 8, 8, 7, 5, 8, 7, 8, 9, 7, 4, 7, 0, 7, 3, 3, 2, 9, 8, 1, 8, 2, 7, 2, 4, 9, 6, 3, 4, 4, 4, 8, 9, 4, 6, 7, 9, 1, 9, 5, 2, 2, 6, 1, 4, 0, 2, 9, 6, 7, 9, 6, 8, 4, 8, 4, 3, 9, 2, 3, 4, 0, 0, 3, 8, 6, 1, 0, 0, 9, 9, 8, 0, 1, 6, 0, 2, 3, 9, 6, 9, 2, 8, 4, 5, 6, 5, 9, 0, 7, 5, 1, 9, 6, 5, 9, 0, 5, 1, 2, 3, 1, 8, 4, 2, 8, 3, 3, 8, 5, 8, 9, 2, 0, 9, 4, 8, 8, 0, 7, 6, 6, 8, 3, 1, 4, 7, 3, 7, 6, 2, 6, 3, 3, 1, 7, 2, 3, 9, 3, 7, 1, 7, 3, 0, 7, 2, 2, 7, 1, 1, 9, 4, 2, 0, 9, 4, 8, 6, 1, 5, 4, 9, 1, 4, 4, 4, 6, 0, 9, 1, 3, 1, 1, 1, 4, 5, 6, 8, 8, 5, 5, 6, 0, 2, 3, 9, 3, 0, 1, 4, 3, 9, 5, 9, 8, 0, 8, 1, 1, 6, 7, 1, 3, 1, 6, 7, 4, 1, 5, 1, 0, 3, 7, 8, 0, 9, 8, 4, 4, 8, 2, 8, 7, 1, 8, 4, 7, 0, 6, 0, 4, 6, 4, 7, 5, 6, 8, 5, 9, 9, 4, 5, 2, 6, 3, 2, 0, 0, 1, 3, 0, 0, 0, 1, 3, 2, 9, 3, 8, 1, 8, 6, 5, 6, 2, 7, 1, 8, 5, 4, 1, 9, 2, 7, 7, 0, 6, 9, 3, 6, 5, 0, 6, 1, 0, 9, 0, 4, 4, 8, 5, 9, 9, 2, 1, 4, 9, 8, 0, 8, 4, 7, 0, 0, 4, 9, 4, 6, 1, 1, 0, 5, 6, 3, 4, 5, 6, 0, 9, 6, 3, 5, 7, 3, 3, 7, 3, 4, 3, 2, 5, 6, 7, 0, 7, 1, 0, 5, 8, 0, 3, 8, 4, 6, 5, 1, 9, 2, 3, 9, 9, 1, 4, 6, 0, 0, 4, 9, 8, 5, 5, 5, 5, 6, 0, 5, 0, 9, 6, 1, 1, 9, 6, 7, 2, 2, 8, 1, 7, 1, 8, 3, 8, 4, 8, 6, 2, 2, 4, 1, 1, 6, 1, 2, 1, 4, 3, 0, 4, 9, 3, 3, 2, 4, 7, 1, 4, 1, 2, 4, 2, 7, 3, 5, 8, 6, 8, 2, 0, 2, 6, 3, 1, 2, 3, 7, 0, 1, 1, 5, 7, 0, 5, 1, 8, 6, 0, 9, 8, 6, 1, 9, 3, 3, 7, 0, 5, 0, 1, 6, 9, 3, 8, 3, 2, 6, 7, 9, 7, 9, 3, 6, 8, 0, 3, 3, 0, 9, 6, 4, 4, 6, 5, 9, 2, 6, 4, 2, 7, 1, 5, 9, 1, 8, 1, 0, 6, 9, 7, 1, 6, 9, 9, 8, 4, 2, 8, 2, 5, 5, 5, 6, 0, 6, 7, 2, 8, 3, 7, 6, 9, 8, 1, 2, 5, 8, 1, 9, 2, 0, 7, 6, 6, 6, 5, 2, 9, 8, 5, 9, 5, 4, 8, 8, 3, 8, 6, 5, 4, 8, 9, 3, 6, 2, 0, 3, 2, 4, 2, 4, 8, 7, 8, 1, 5, 7, 2, 0, 0, 9, 2, 6, 5, 1, 4, 7, 0, 6, 9, 3, 5, 9, 3, 2, 1, 8, 2, 4, 9, 3, 5, 4, 4, 9, 5, 2, 5, 7, 5, 9, 0, 3, 9, 7, 0, 9, 1, 7, 2, 2, 4, 2, 1, 9, 3, 0, 8, 7, 8, 4, 3, 1, 0, 1, 8, 9, 8, 2, 1, 4, 7, 6, 4, 2, 2, 9, 3, 3, 3, 5, 6, 8, 3, 0, 8, 4, 8, 3, 9, 8, 1, 7, 0, 3, 5, 2, 4, 4, 5, 3, 3, 7, 1, 3, 0, 1, 0, 4, 5, 5, 7, 0, 1, 4, 4, 5, 7, 4, 5, 6, 6, 5, 0, 1, 1, 1, 8, 7, 5, 7, 3, 5, 2, 9, 2, 2, 4, 7, 6, 4, 7, 2, 8, 6, 2, 4, 3, 6, 5, 3, 8, 3, 2, 7, 3, 1, 9, 4, 9, 0, 6, 6, 6, 4, 1, 5, 1, 0, 7, 4, 7, 3, 0, 5, 2, 0, 3, 9, 0, 0, 3, 2, 0, 8, 2, 6, 7, 6, 1, 4, 9, 0, 7, 0, 5, 6, 9, 3, 3, 1, 6, 9, 1, 7, 7, 8, 1, 3, 5, 1, 2, 6, 2, 3, 9, 7, 1, 1, 8, 1, 7, 5, 0, 8, 4, 9, 7, 8, 3, 2, 7, 0, 2, 0, 7, 5, 3, 1, 7, 4, 8, 3, 0, 5, 8, 5, 0, 3, 7, 2, 7, 0, 3, 7, 5, 8, 9, 2, 8, 7, 0, 8, 4, 3, 0, 0, 5, 7, 8, 0, 1, 6, 1, 1, 1, 8, 1, 6, 2, 5, 5, 6, 1, 3, 4, 0, 7, 6, 7, 4, 9, 6, 0, 9, 3, 7, 8, 7, 9, 5, 2, 0, 7, 6, 2, 6, 8, 5, 5, 6, 7, 0, 5, 1, 5, 9, 3, 9, 9, 8, 8, 3, 0, 4, 3, 0, 9, 7, 5, 2, 2, 5, 6, 8, 4, 7, 5, 1, 4, 4, 1, 2, 9, 4, 9, 2, 6, 6, 2, 9, 5, 9, 2, 1, 8, 9, 8, 5, 4, 9, 5, 2, 9, 9, 0, 9, 0, 5, 0, 5, 1, 5, 8, 2, 8, 4, 9, 2, 9, 1, 1, 4, 6, 3, 7, 4, 8, 5, 0, 9, 6, 2, 3, 4, 3, 3, 7, 0, 5, 6, 9, 4, 1, 2, 4, 1, 4, 3, 2, 4, 5, 9, 3, 1, 9, 1, 7, 7, 0, 0, 5, 2, 5, 4, 3, 6, 4, 3, 5, 1, 5, 4, 0, 1, 1, 5, 9, 9, 1, 4, 8, 0, 4, 5, 5, 4, 6, 6, 6, 3, 4, 8, 4, 2, 1, 0, 4, 2, 9, 5, 7, 1, 7, 2, 6, 5, 8, 3, 0, 4, 5, 3, 2, 8, 8, 4, 7, 2, 4, 7, 4, 4, 0, 3, 0, 0, 0, 2, 2, 0, 2, 4, 4, 5, 2, 0, 4, 7, 1, 7, 3, 2, 7, 6, 2, 6, 5, 9, 5, 8, 5, 2, 6, 0, 3, 6, 3, 9, 5, 9, 8, 1, 7, 7, 6, 1, 1, 2, 8, 6, 4, 5, 7, 5, 2, 9, 7, 1, 1, 8, 2, 7, 5, 0, 5, 8, 8, 8, 7, 0, 9, 4, 3, 2, 7, 3, 5, 9, 9, 1, 0, 0, 3, 6, 6, 9, 1, 0, 3, 7, 9, 6, 2, 6, 7, 2, 2, 2, 9, 3, 6, 0, 0, 3, 9, 5, 5, 8, 2, 9, 3, 0, 9, 3, 3, 4, 8, 2, 4, 7, 8, 6, 6, 7, 2, 5, 1, 2, 7, 6, 3, 3, 6, 8, 4, 8, 9, 3, 7, 6, 3, 0, 9, 7, 9, 8, 1, 2, 9, 1, 0, 9, 0, 6, 7, 9, 7, 9, 4, 1, 6, 3, 3, 8, 8, 3, 3, 9, 5, 9, 6, 7, 4, 0, 6, 6, 5, 0, 8, 2, 0, 2, 2, 8, 6, 2, 3, 7, 9, 3, 7, 2, 0, 6, 2, 9, 5, 3, 8, 2, 3, 2, 7, 0, 4, 5, 1, 5, 7, 8, 7, 0, 4, 2, 0, 6, 6, 9, 2, 6, 9, 1, 0, 6, 3, 0, 4, 0, 7, 3, 7, 6, 6, 2, 7, 1, 4, 5, 2, 9, 9, 0, 7, 0, 6, 7, 3, 2, 0, 9, 3, 9, 3, 9, 9, 9, 2, 1, 5, 6, 8, 7, 5, 5, 3, 6, 6, 3, 3, 1, 6, 5, 4, 2, 5, 5, 5, 1, 2, 8, 2, 7, 1, 4, 1, 3, 4, 2, 5, 7, 8, 1, 7, 7, 4, 3, 4, 4, 0, 7, 3, 0, 4, 6, 3, 0, 1, 5, 6, 6, 7, 5, 6, 9, 2, 1, 3, 6, 8, 7, 6, 3, 6, 5, 9, 8, 5, 7, 1, 5, 8, 3, 6, 9, 4, 9, 6, 2, 3, 9, 3, 0, 2, 6, 4, 8, 8, 3, 8, 6, 5, 2, 6, 4, 4, 3, 4, 5, 9, 5, 0, 0, 8, 8, 5, 2, 4, 7, 1, 9, 7, 3, 5, 1, 5, 9, 3, 1, 3, 8, 7, 2, 6, 1, 5, 4, 3, 7, 6, 0, 8, 9, 3, 6, 6, 5, 9, 4, 1, 4, 4, 2, 7, 2, 3, 5, 7, 0, 3, 6, 4, 1, 7, 0, 0, 7, 0, 5, 7, 0, 2, 6, 1, 2, 5, 4, 4, 8, 4, 4, 5, 2, 4, 0, 4, 9, 6, 9, 8, 0, 1, 3, 8, 8, 3, 4, 8, 5, 5, 6, 0, 5, 0, 4, 8, 0, 4, 6, 0, 0, 0, 1, 1, 3, 3, 2, 5, 4, 6, 3, 7, 6, 9, 2, 4, 8, 4, 2, 4, 5, 4, 4, 0, 3, 2, 5, 5, 0, 1, 0, 3, 8, 4, 1, 2, 4, 1, 3, 0, 1, 8, 1, 2, 0, 1, 4, 2, 8, 3, 3, 9, 9, 1, 2, 3, 0, 1, 2, 3, 5, 6, 2, 5, 4, 4, 9, 4, 3, 9, 3, 9, 4, 2, 8, 7, 0, 2, 2, 8, 8, 3, 3, 3, 4, 7, 5, 3, 7, 6, 8, 4, 0, 3, 9, 6, 9, 4, 8, 9, 3, 4, 6, 9, 7, 4, 3, 9, 5, 7, 6, 1, 4, 5, 4, 4, 6, 6, 8, 7, 9, 8, 6, 2, 7, 6, 1, 4, 4, 5, 2, 2, 1, 7, 7, 9, 2, 5, 8, 9, 2, 8, 5, 4, 6, 3, 9, 5, 3, 1, 1, 5, 5, 1, 9, 3, 2, 8, 9, 3, 8, 4, 2, 7, 4, 9, 1, 4, 5, 0, 5, 1, 9, 8, 7, 7, 5, 0, 6, 5, 0, 4, 4, 4, 6, 6, 5, 4, 9, 0, 6, 3, 6, 1, 9, 7, 6, 5, 4, 1, 2, 4, 5, 1, 8, 4, 9, 7, 8, 2, 2, 8, 2, 5, 4, 2, 3, 6, 3, 3, 0, 8, 7, 6, 1, 4, 6, 0, 7, 6, 7, 7, 2, 1, 9, 9, 0, 6, 2, 3, 6, 1, 8, 6, 4, 5, 8, 5, 6, 6, 4, 5, 6, 2, 7, 8, 1, 4, 3, 9, 1, 9, 5, 2, 5, 3, 2, 6, 9, 3, 6, 2, 6, 9, 3, 4, 7, 6, 3, 1, 4, 4, 6, 2, 0, 2, 0, 5, 0, 3, 7, 4, 1, 7, 3, 0, 6, 5, 5, 6, 7, 5, 3, 6, 1, 7, 9, 5, 6, 2, 8, 1, 9, 2, 7, 3, 4, 4, 6, 7, 4, 0, 2, 1, 7, 2, 2, 0, 0, 6, 8, 7, 8, 6, 3, 6, 2, 5, 8, 3, 0, 3, 0, 9, 3, 4, 6, 4, 5, 8, 9, 1, 3, 2, 4, 8, 6, 8, 1, 1, 1, 9, 0, 2, 5, 8, 2, 6, 8, 3, 4, 0, 0, 4, 9, 3, 4, 0, 1, 8, 3, 4, 5, 8, 1, 6, 0, 0, 8, 9, 3, 6, 0, 5, 3, 4, 9, 0, 4, 0, 6, 4, 7, 8, 1, 5, 3, 2, 4, 2, 5, 3, 0, 9, 3, 3, 9, 1, 0, 6, 0, 6, 4, 9, 2, 6, 0, 4, 8, 4, 1, 8, 6, 2, 6, 0, 3, 3, 0, 5, 6, 8, 1, 7, 0, 1, 8, 1, 9, 6, 4, 0, 6, 3, 4, 6, 3, 8, 5, 0, 7, 7, 1, 5, 8, 7, 2, 1, 1, 5, 8, 3, 2, 6, 1, 8, 3, 0, 1, 5, 2, 3, 3, 7, 2, 3, 1, 3, 1, 4, 2, 3, 2, 5, 0, 2, 2, 9, 1, 9, 2, 7, 5, 1, 0, 0, 2, 8, 1, 6, 4, 8, 1, 2, 0, 7, 6, 7, 6, 9, 9, 0, 6, 4, 3, 4, 4, 0, 0, 8, 3, 4, 6, 7, 9, 1, 6, 9, 3, 9, 6, 5, 3, 5, 4, 8, 7, 6, 4, 5, 8, 1, 6, 4, 2, 1, 9, 2, 9, 2, 1, 6, 2, 8, 3, 6, 0, 5, 1, 4, 2, 9, 5, 6, 3, 7, 6, 1, 7, 2, 6, 3, 4, 0, 9, 8, 8, 6, 9, 1, 1, 7, 3, 8, 7, 9, 6, 2, 0, 1, 9, 8, 2, 1, 7, 4, 3, 8, 1, 7, 3, 1, 2, 1, 3, 4, 0, 5, 5, 2, 7, 7, 9, 3, 8, 0, 8, 0, 1, 5, 1, 4, 0, 1, 5, 5, 8, 6, 2, 1, 3, 6, 7, 2, 5, 7, 6, 0, 2, 9, 0, 5, 7, 1, 3, 4, 1, 1, 1, 9, 0, 5, 4, 2, 4, 1, 6, 8, 3, 1, 6, 9, 5, 7, 2, 4, 3, 4, 0, 5, 9, 2, 7, 7, 6, 7, 2, 2, 4, 9, 3, 9, 1, 0, 2, 1, 0, 0, 5, 9, 1, 8, 0, 4, 1, 5, 6, 4, 7, 6, 7, 5, 9, 3, 6, 2, 0, 1, 2, 9, 4, 2, 4, 4, 2, 1, 9, 8, 3, 0, 1, 0, 2, 9, 9, 7, 2, 9, 8, 7, 9, 5, 8, 3, 5, 6, 7, 1, 4, 4, 7, 0, 6, 2, 7, 9, 9, 8, 9, 8, 0, 9, 9, 8, 8, 4, 3, 2, 7, 5, 7, 0, 1, 6, 1, 8, 0, 8, 2, 5, 9, 1, 9, 1, 3, 2, 8, 4, 7, 3, 3, 2, 0, 0, 2, 1, 1, 1, 0, 6, 7, 6, 3, 8, 0, 6, 2, 4, 2, 1, 5, 5, 8, 6, 9, 8, 0, 5, 1, 9, 8, 5, 3, 0, 5, 3, 4, 0, 4, 1, 6, 2, 0, 9, 6, 7, 3, 9, 9, 7, 9, 6, 7, 1, 2, 0, 2, 6, 3, 3, 2, 9, 7, 4, 8, 6, 1, 1, 7, 0, 5, 3, 1, 8, 0, 7, 7, 1, 4, 1, 9, 2, 3, 6, 0, 0, 4, 1, 8, 2, 6, 5, 8, 0, 3, 3, 8, 8, 3, 6, 2, 2, 2, 4, 3, 9, 4, 9, 1, 2, 2, 7, 6, 2, 5, 0, 6, 8, 7, 0, 4, 5, 7, 1, 4, 1, 2, 9, 9, 7, 2, 5, 5, 9, 7, 1, 6, 8, 0, 4, 1, 1, 0, 2, 9, 9, 6, 1, 1, 4, 7, 0, 8, 8, 2, 3, 3, 2, 0, 9, 8, 7, 8, 4, 1, 0, 5, 4, 2, 4, 4, 4, 3, 6, 4, 6, 7, 3, 7, 5, 6, 8, 5, 1, 8, 8, 1, 5, 3, 9, 4, 1, 3, 9, 3, 3, 2, 3, 1, 3, 9, 6, 1, 2, 5, 6, 9, 8, 5, 0, 6, 2, 6, 2, 8, 2, 4, 8, 1, 3, 9, 2, 9, 4, 2, 5, 3, 7, 9, 3, 1, 5, 7, 0, 9, 8, 4, 5, 7, 6, 9, 5, 2, 6, 4, 0, 4, 3, 4, 0, 0, 9, 2, 4, 3, 9, 5, 3, 0, 3, 5, 1, 8, 9, 6, 4, 9, 3, 6, 1, 7, 5, 7, 1, 3, 8, 4, 0, 6, 9, 9, 2, 9, 0, 2, 4, 4, 2, 1, 7, 2, 4, 7, 7, 5, 4, 1, 0, 4, 9, 8, 4, 2, 1, 2, 7, 2, 6, 6, 1, 2, 3, 6, 3, 3, 5, 6, 6, 2, 3, 8, 5, 6, 0, 5, 9, 8, 2, 3, 6, 2, 9, 7, 9, 4, 1, 3, 7, 4, 7, 4, 3, 0, 5, 9, 7, 6, 0, 5, 9, 3, 7, 2, 8, 9, 7, 3, 5, 4, 8, 3, 4, 6, 4, 6, 7, 4, 9, 0, 3, 7, 4, 1, 7, 9, 8, 8, 5, 6, 2, 6, 5, 1, 9, 1, 4, 8, 7, 4, 3, 5, 5, 5, 2, 8, 6, 2, 8, 9, 5, 0, 5, 3, 4, 0, 2, 4, 7, 6, 7, 0, 3, 9, 9, 2, 9, 6, 5, 5, 9, 1, 7, 3, 3, 0, 3, 6, 4, 9, 9, 0, 4, 3, 9, 7, 5, 3, 5, 3, 8, 1, 6, 3, 0, 5, 9, 7, 3, 7, 5, 7, 0, 6, 6, 6, 3, 1, 4, 0, 3, 9, 7, 0, 6, 5, 8, 7, 6, 3, 8, 2, 6, 6, 8, 3, 2, 8, 7, 3, 0, 3, 1, 7, 9, 1, 3, 5, 9, 0, 4, 6, 8, 3, 3, 2, 9, 1, 7, 0, 5, 2, 4, 0, 5, 4, 5, 2, 2, 9, 5, 2, 8, 1, 8, 7, 4, 1, 5, 2, 6, 1, 7, 1, 0, 7, 0, 5, 4, 0, 7, 8, 7, 4, 1, 4, 3, 4, 5, 9, 1, 3, 8, 8, 9, 6, 7, 3, 2, 1, 7, 9, 2, 3, 9, 9, 2, 7, 2, 1, 7, 2, 6, 8, 6, 2, 4, 0, 1, 7, 3, 8, 6, 4, 7, 6, 7, 2, 2, 3, 2, 4, 6, 6, 6, 2, 2, 5, 8, 6, 2, 8, 9, 2, 0, 2, 7, 1, 0, 9, 3, 5, 1, 0, 1, 8, 4, 3, 2, 8, 8, 9, 5, 1, 6, 8, 8, 1, 2, 5, 6, 1, 7, 2, 8, 7, 7, 0, 1, 3, 7, 4, 3, 2, 2, 5, 7, 1, 8, 1, 3, 2, 9, 6, 9, 1, 9, 2, 6, 2, 3, 4, 0, 3, 0, 9, 9, 1, 7, 8, 3, 2, 8, 1, 6, 6, 0, 9, 0, 9, 1, 9, 3, 5, 7, 9, 2, 1, 5, 4, 0, 0, 3, 6, 0, 3, 3, 8, 7, 3, 4, 9, 8, 7, 9, 9, 2, 9, 9, 8, 3, 8, 3, 0, 1, 8, 9, 5, 9, 9, 6, 1, 5, 1, 8, 0, 7, 8, 9, 2, 7, 1, 8, 1, 2, 9, 2, 7, 5, 5, 2, 2, 9, 7, 1, 0, 6, 3, 8, 2, 2, 1, 1, 0, 7, 9, 3, 3, 0, 5, 4, 3, 5, 2, 2, 3, 6, 0, 6, 1, 8, 0, 5, 0, 1, 6, 4, 2, 9, 3, 3, 8, 2, 6, 8, 6, 7, 4, 1, 2, 5, 9, 1, 9, 0, 7, 3, 0, 1, 5, 3, 8, 1, 1, 1, 1, 6, 1, 4, 1, 3, 5, 0, 7, 9, 8, 6, 9, 3, 1, 5, 9, 1, 9, 5, 5, 1, 3, 4, 1, 1, 6, 2, 2, 6, 6, 6, 6, 1, 9, 5, 7, 8, 1, 3, 1, 8, 7, 7, 2, 0, 7, 9, 1, 8, 8, 7, 8, 3, 4, 1, 0, 0, 4, 0, 7, 6, 9, 9, 2, 2, 6, 2, 3, 5, 2, 6, 1, 1, 0, 1, 0, 3, 8, 0, 6, 8, 0, 6, 9, 5, 4, 6, 9, 4, 6, 9, 7, 9, 8, 8, 4, 7, 2, 7, 2, 8, 3, 6, 2, 5, 8, 4, 8, 8, 8, 8, 6, 8, 2, 7, 0, 3, 6, 6, 8, 8, 2, 6, 3, 0, 3, 3, 9, 5, 7, 8, 5, 4, 9, 7, 6, 4, 3, 8, 6, 8, 6, 7, 3, 7, 5, 7, 3, 8, 3, 4, 4, 3, 0, 7, 0, 5, 1, 8, 9, 0, 5, 7, 0, 0, 3, 5, 5, 4, 8, 3, 4, 3, 9, 8, 0, 1, 1, 8, 7, 2, 9, 0, 2, 4, 8, 1, 9, 1, 6, 1, 6, 1, 0, 4, 0, 3, 4, 6, 5, 0, 7, 7, 9, 7, 5, 9, 6, 3, 7, 8, 3, 9, 0, 8, 2, 6, 0, 4, 0, 2, 5, 2, 3, 3, 3, 4, 3, 7, 0, 8, 4, 7, 6, 2, 3, 2, 9, 3, 5, 7, 9, 2, 3, 1, 9, 4, 6, 3, 7, 7, 6, 1, 2, 2, 4, 9, 1, 0, 3, 1, 2, 0, 3, 8, 1, 2, 2, 0, 6, 4, 6, 1, 5, 0, 3, 1, 3, 1, 7, 0, 0, 3, 6, 2, 6, 2, 8, 7, 5, 8, 1, 8, 7, 2, 9, 8, 9, 4, 6, 6, 5, 0, 4, 9, 9, 8, 6, 4, 2, 1, 1, 8, 5, 1, 5, 0, 9, 9, 6, 0, 1, 5, 7, 3, 5, 8, 3, 1, 1, 0, 6, 5, 0, 5, 9, 5, 8, 5, 0, 6, 6, 8, 0, 6, 6, 3, 6, 1, 8, 1, 3, 9, 5, 0, 4, 6, 2, 5, 1, 9, 3, 9, 2, 0, 2, 7, 1, 1, 0, 0, 7, 9, 1, 9, 7, 4, 3, 1, 9, 2, 9, 2, 0, 6, 3, 4, 7, 2, 6, 0, 8, 1, 3, 4, 4, 6, 6, 2, 4, 1, 4, 4, 3, 6, 3, 5, 9, 4, 0, 1, 2, 3, 1, 6, 3, 1, 1, 2, 1, 6, 6, 2, 9, 8, 5, 4, 8, 5, 0, 2, 8, 9, 3, 9, 5, 9, 8, 5, 4, 9, 2, 1, 4, 6, 9, 1, 8, 2, 4, 3, 5, 2, 8, 6, 1, 5, 9, 7, 5, 8, 8, 4, 0, 2, 8, 7, 4, 1, 9, 3, 9, 0, 0, 6, 7, 2, 2, 6, 7, 2, 9, 3, 1, 1, 5, 1, 2, 7, 5, 7, 8, 4, 6, 9, 1, 4, 9, 9, 5, 4, 1, 1, 8, 4, 5, 3, 6, 5, 0, 0, 9, 6, 9, 6, 5, 6, 8, 9, 5, 2, 2, 8, 2, 8, 7, 3, 3, 9, 3, 5, 5, 0, 9, 4, 6, 9, 5, 5, 7, 3, 0, 5, 6, 6, 0, 0, 4, 2, 4, 5, 0, 1, 8, 2, 9, 4, 0, 8, 4, 0, 8, 0, 8, 4, 1, 7, 9, 0, 5, 4, 1, 9, 0, 7, 3, 3, 5, 4, 1, 8, 1, 0, 2, 1, 6, 7, 6, 8, 9, 0, 2, 9, 3, 2, 5, 8, 9, 7, 2, 9, 4, 3, 6, 1, 2, 8, 6, 2, 3, 8, 0, 7, 6, 4, 2, 7, 9, 4, 0, 8, 6, 7, 1, 5, 1, 7, 7, 5, 0, 7, 7, 2, 1, 2, 6, 1, 2, 8, 6, 1, 3, 8, 9, 2, 2, 1, 6, 8, 4, 3, 9, 8, 1, 0, 8, 4, 7, 7, 9, 6, 3, 7, 6, 1, 5, 0, 7, 8, 2, 6, 7, 0, 0, 6, 7, 9, 7, 4, 8, 3, 7, 1, 1, 7, 8, 2, 4, 5, 1, 9, 7, 2, 7, 7, 0, 5, 7, 8, 0, 6, 0, 2, 2, 1, 5, 3, 7, 8, 1, 6, 8, 1, 1, 2, 5, 8, 2, 1, 9, 7, 0, 5, 3, 9, 0, 3, 5, 3, 1, 0, 9, 5, 0, 5, 9, 0, 9, 5, 6, 5, 0, 4, 9, 9, 8, 9, 7, 5, 1, 4, 7, 8, 5, 1, 7, 3, 2, 9, 9, 6, 9, 8, 6, 4, 0, 6, 6, 8, 4, 2, 1, 7, 4, 0, 2, 1, 0, 8, 9, 8, 1, 3, 9, 7, 8, 2, 2, 2, 3, 9, 4, 3, 6, 1, 9, 4, 4, 3, 1, 4, 9, 8, 1, 2, 0, 6, 6, 3, 6, 1, 9, 1, 5, 6, 2, 9, 4, 7, 0, 9, 4, 0, 0, 5, 5, 2, 1, 1, 8, 5, 4, 7, 7, 7, 2, 1, 2, 6, 3, 4, 0, 5, 7, 3, 7, 2, 7, 7, 3, 7, 5, 1, 0, 0, 8, 8, 7, 0, 4, 4, 3, 2, 1, 2, 1, 2, 2, 7, 8, 3, 7, 6, 5, 6, 2, 8, 5, 0, 2, 6, 7, 4, 3, 6, 1, 9, 0, 6, 6, 9, 4, 2, 0, 7, 4, 7, 5, 1, 0, 1, 0, 4, 9, 8, 1, 2, 1, 6, 3, 7, 9, 4, 2, 3, 6, 4, 2, 1, 3, 9, 1, 1, 6, 5, 3, 1, 6, 2, 5, 3, 8, 4, 4, 8, 2, 5, 2, 7, 4, 8, 9, 9, 6, 5, 4, 5, 0, 0, 8, 2, 2, 6, 4, 6, 6, 9, 7, 1, 9, 4, 8, 3, 8, 6, 8, 6, 0, 2, 3, 1, 4, 1, 3, 7, 1, 9, 6, 6, 1, 8, 4, 9, 9, 1, 5, 8, 5, 9, 5, 0, 4, 3, 0, 9, 2, 5, 6, 0, 8, 6, 8, 8, 7, 4, 7, 1, 5, 8, 3, 2, 3, 6, 2, 2, 7, 0, 8, 6, 7, 6, 4, 0, 4, 6, 0, 1, 1, 9, 6, 3, 1, 0, 2, 6, 5, 4, 3, 9, 1, 5, 6, 2, 3, 6, 5, 5, 3, 4, 0, 0, 0, 8, 8, 8, 4, 7, 4, 5, 4, 6, 5, 6, 7, 4, 2, 0, 2, 0, 9, 2, 9, 0, 8, 5, 8, 9, 0, 2, 1, 3, 2, 6, 7, 9, 9, 8, 7, 9, 9, 7, 0, 4, 9, 1, 6, 7, 5, 2, 4, 1, 0, 6, 6, 9, 0, 4, 9, 9, 1, 6, 1, 5, 9, 7, 3, 2, 9, 6, 2, 8, 0, 6, 9, 6, 4, 1, 0, 1, 1, 3, 0, 8, 9, 0, 9, 9, 4, 8, 0, 2, 3, 4, 1, 4, 7, 8, 0, 5, 8, 8, 2, 5, 0, 2, 3, 4, 2, 1, 2, 1, 3, 6, 7, 4, 5, 5, 5, 9, 9, 5, 5, 2, 1, 9, 6, 2, 4, 8, 6, 7, 7, 8, 7, 9, 2, 8, 5, 4, 3, 8, 9, 6, 5, 7, 6, 1, 3, 1, 4, 6, 1, 3, 8, 5, 2, 1, 6, 3, 3, 4, 2, 2, 9, 6, 7, 6, 0, 0, 7, 8, 5, 6, 2, 6, 6, 7, 3, 4, 5, 2, 8, 1, 2, 7, 1, 5, 7, 2, 8, 0, 5, 1, 3, 4, 6, 6, 9, 2, 1, 6, 4, 1, 9, 6, 0, 6, 8, 4, 8, 2, 9, 7, 0, 0, 3, 7, 6, 1, 7, 5, 5, 5, 9, 2, 2, 0, 6, 3, 5, 4, 4, 8, 2, 1, 6, 3, 8, 8, 6, 6, 9, 2, 1, 8, 9, 4, 1, 5, 4, 1, 6, 5, 1, 5, 6, 6, 7, 4, 1, 2, 8, 5, 9, 1, 7, 6, 8, 1, 3, 4, 1, 1, 2, 6, 2, 8, 5, 6, 0, 6, 6, 7, 2, 6, 9, 8, 2, 2, 3, 8, 6, 9, 5, 0, 9, 5, 8, 2, 6, 7, 6, 2, 8, 9, 3, 0, 0, 1, 2, 2, 9, 2, 5, 0, 9, 5, 9, 5, 5, 1, 1, 1, 3, 1, 5, 3, 0, 9, 6, 2, 6, 2, 7, 0, 8, 7, 6, 4, 3, 5, 3, 0, 4, 9, 3, 9, 1, 5, 9, 1, 2, 3, 3, 2, 1, 1, 3, 9, 4, 5, 5, 9, 6, 9, 3, 5, 1, 4, 4, 6, 3, 8, 8, 3, 5, 9, 2, 4, 4, 3, 0, 5, 6, 3, 4, 2, 1, 3, 0, 8, 1, 6, 4, 0, 5, 7, 7, 3, 3, 2, 4, 2, 3, 9, 4, 9, 4, 0, 5, 4, 4, 8, 7, 0, 4, 7, 4, 9, 4, 7, 3, 0, 0, 9, 0, 1, 1, 5, 7, 1, 7, 4, 7, 4, 4, 5, 4, 0, 5, 0, 7, 6, 7, 2, 7, 1, 9, 0, 8, 0, 3, 6, 0, 0, 3, 8, 7, 0, 6, 2, 2, 0, 8, 8, 0, 0, 2, 4, 6, 8, 0, 8, 8, 3, 4, 5, 4, 2, 3, 2, 7, 2, 7, 7, 6, 0, 8, 8, 3, 5, 8, 6, 5, 7, 6, 3, 2, 8, 2, 9, 6, 5, 1, 1, 3, 6, 0, 2, 8, 9, 9, 3, 7, 8, 2, 5, 5, 3, 1, 8, 8, 7, 8, 0, 1, 5, 5, 3, 5, 2, 1, 4, 8, 1, 1, 3, 3, 6, 1, 0, 0, 5, 1, 6, 5, 8, 7, 5, 5, 0, 9, 3, 3, 0, 6, 4, 4, 9, 8, 0, 4, 2, 4, 2, 1, 0, 0, 4, 1, 9, 6, 1, 3, 1, 8, 5, 3, 9, 4, 1, 2, 9, 6, 7, 9, 9, 3, 7, 2, 9, 3, 7, 1, 7, 0, 8, 8, 8, 7, 6, 8, 4, 1, 9, 6, 0, 6, 8, 4, 9, 2, 1, 2, 8, 8, 3, 9, 1, 3, 3, 2, 9, 1, 0, 2, 3, 3, 2, 4, 6, 4, 2, 7, 9, 9, 5, 4, 0, 2, 9, 7, 5, 9, 2, 6, 0, 6, 1, 3, 7, 5, 7, 2, 8, 4, 8, 6, 6, 6, 5, 4, 6, 7, 4, 3, 7, 4, 8, 1, 3, 4, 0, 5, 0, 6, 9, 4, 0, 3, 7, 7, 2, 8, 9, 9, 4, 5, 8, 7, 9, 2, 7, 7, 5, 1, 4, 8, 5, 6, 8, 9, 0, 4, 1, 0, 5, 5, 6, 1, 4, 6, 6, 5, 8, 4, 7, 5, 4, 8, 5, 2, 7, 1, 1, 2, 4, 7, 6, 8, 9, 4, 1, 6, 0, 9, 0, 0, 1, 6, 6, 0, 1, 9, 5, 8, 4, 2, 8, 4, 2, 0, 8, 9, 3, 0, 8, 1, 1, 6, 5, 2, 4, 3, 2, 2, 0, 4, 4, 0, 4, 4, 7, 7, 4, 0, 7, 5, 3, 2, 8, 8, 0, 1, 3, 7, 8, 4, 7, 4, 3, 4, 3, 8, 0, 8, 5, 2, 0, 5, 7, 7, 5, 6, 6, 5, 1, 9, 7, 1, 2, 1, 5, 0, 6, 9, 6, 8, 0, 6, 7, 1, 6, 3, 6, 3, 9, 7, 6, 4, 3, 9, 5, 9, 3, 6, 9, 7, 4, 1, 1, 4, 5, 2, 9, 6, 6, 1, 2, 4, 4, 1, 6, 4, 1, 4, 4, 4, 9, 9, 7, 8, 5, 3, 3, 6, 5, 7, 9, 6, 4, 0, 2, 5, 1, 4, 9, 6, 0, 4, 4, 2, 5, 6, 5, 6, 4, 3, 9, 4, 8, 2, 1, 3, 0, 1, 7, 0, 6, 8, 5, 7, 2, 3, 6, 5, 8, 8, 7, 7, 8, 0, 9, 4, 5, 4, 8, 8, 9, 6, 0, 8, 2, 8, 8, 7, 6, 2, 3, 2, 9, 6, 9, 3, 1, 0, 3, 1, 9, 9, 6, 0, 9, 1, 5, 7, 7, 1, 9, 9, 9, 5, 2, 0, 2, 7, 4, 5, 7, 4, 5, 5, 5, 8, 7, 1, 3, 4, 2, 1, 0, 7, 7, 3, 6, 1, 3, 5, 8, 7, 9, 6, 9, 5, 3, 5, 6, 2, 7, 7, 8, 7, 0, 2, 4, 9, 4, 2, 4, 4, 6, 0, 1, 1, 9, 1, 3, 4, 8, 1, 8, 8, 0, 3, 3, 2, 5, 2, 5, 4, 3, 8, 7, 2, 7, 7, 4, 8, 2, 0, 8, 5, 5, 3, 6, 8, 0, 9, 4, 1, 7, 1, 4, 4, 8, 8, 0, 6, 3, 5, 5, 8, 4, 1, 6, 8, 5, 0, 3, 3, 2, 3, 6, 3, 0, 3, 3, 6, 2, 5, 6, 1, 2, 1, 6, 7, 3, 7, 9, 7, 1, 6, 8, 8, 2, 9, 3, 9, 0, 5, 6, 6, 3, 2, 8, 6, 1, 6, 3, 3, 9, 8, 8, 1, 0, 7, 7, 4, 8, 7, 5, 1, 6, 2, 5, 5, 8, 8, 5, 4, 6, 0, 9, 6, 0, 1, 6, 8, 3, 9, 8, 0, 5, 2, 6, 6, 2, 7, 2, 1, 9, 8, 4, 5, 1, 9, 2, 5, 6, 0, 8, 9, 1, 9, 2, 8, 3, 8, 1, 1, 9, 6, 8, 5, 6, 8, 3, 3, 7, 5, 0, 8, 1, 2, 2, 6, 6, 0, 1, 4, 7, 3, 5, 2, 9, 1, 8, 5, 6, 5, 5, 5, 9, 3, 3, 2, 5, 2, 5, 2, 6, 8, 9, 7, 5, 6, 4, 7, 5, 0, 4, 4, 1, 5, 0, 0, 7, 0, 2, 7, 9, 2, 6, 7, 3, 8, 5, 3, 5, 8, 9, 6, 9, 9, 0, 7, 0, 5, 6, 9, 3, 7, 6, 9, 0, 9, 7, 9, 9, 9, 6, 8, 2, 5, 1, 1, 8, 7, 7, 1, 3, 3, 7, 5, 0, 0, 7, 3, 4, 0, 7, 0, 7, 6, 5, 6, 5, 3, 1, 7, 9, 3, 4, 9, 8, 9, 9, 5, 9, 2, 4, 6, 7, 5, 9, 4, 2, 3, 2, 7, 8, 4, 5, 6, 9, 9, 7, 4, 3, 6, 2, 1, 7, 2, 8, 7, 0, 3, 1, 1, 0, 9, 4, 7, 5, 3, 3, 7, 4, 5, 2, 0, 4, 3, 1, 3, 3, 9, 2, 1, 2, 8, 3, 1, 6, 4, 0, 2, 5, 3, 0, 1, 0, 2, 4, 8, 3, 1, 4, 7, 4, 3, 2, 6, 3, 4, 1, 0, 4, 3, 1, 3, 8, 1, 5, 9, 5, 4, 9, 6, 2, 7, 7, 0, 8, 4, 8, 3, 0, 0, 0, 1, 5, 7, 3, 4, 2, 0, 8, 0, 0, 2, 7, 0, 9, 0, 7, 5, 6, 5, 5, 6, 0, 7, 2, 7, 7, 5, 6, 9, 0, 1, 8, 5, 6, 1, 9, 1, 3, 4, 3, 2, 4, 1, 9, 1, 2, 8, 9, 1, 9, 6, 2, 9, 2, 8, 4, 6, 2, 2, 9, 9, 5, 5, 6, 0, 2, 2, 8, 1, 3, 1, 5, 6, 7, 2, 6, 2, 1, 0, 2, 4, 8, 3, 0, 7, 2, 6, 3, 2, 8, 4, 2, 1, 7, 9, 2, 0, 3, 3, 9, 3, 3, 1, 9, 9, 2, 0, 3, 6, 0, 4, 1, 8, 4, 9, 3, 7, 2, 4, 5, 5, 7, 4, 9, 3, 4, 6, 1, 3, 0, 1, 9, 3, 5, 6, 0, 9, 4, 1, 1, 9, 4, 1, 2, 6, 3, 2, 9, 1, 0, 2, 8, 9, 1, 5, 7, 6, 2, 2, 1, 8, 8, 6, 2, 4, 2, 4, 9, 9, 9, 7, 2, 2, 1, 9, 6, 9, 0, 9, 0, 1, 6, 5, 1, 0, 7, 2, 3, 5, 3, 3, 5, 9, 2, 7, 0, 6, 3, 4, 4, 4, 0, 3, 7, 5, 0, 5, 0, 4, 5, 1, 7, 2, 9, 5, 2, 2, 4, 2, 2, 0, 0, 9, 2, 7, 6, 9, 0, 0, 3, 9, 4, 9, 1, 5, 3, 7, 3, 6, 1, 1, 2, 3, 8, 9, 8, 2, 5, 4, 9, 3, 9, 9, 3, 7, 4, 4, 5, 1, 0, 6, 2, 2, 1, 8, 8, 7, 1, 5, 6, 5, 1, 6, 4, 0, 5, 1, 8, 9, 4, 1, 6, 5, 5, 5, 7, 3, 3, 4, 5, 3, 0, 3, 6, 5, 1, 7, 4, 0, 9, 9, 7, 2, 6, 7, 7, 3, 4, 7, 7, 7, 6, 5, 5, 6, 2, 7, 2, 4, 0, 6, 5, 5, 5, 8, 7, 4, 3, 0, 2, 8, 5, 0, 1, 2, 6, 6, 4, 9, 1, 9, 9, 9, 5, 2, 1, 7, 8, 8, 4, 6, 9, 7, 9, 4, 3, 4, 8, 9, 6, 7, 3, 3, 8, 7, 3, 1, 6, 7, 4, 5, 3, 5, 0, 8, 5, 4, 4, 3, 5, 5, 5, 6, 6, 6, 2, 5, 5, 2, 8, 1, 9, 1, 5, 8, 3, 7, 8, 6, 6, 2, 1, 0, 0, 0, 2, 7, 4, 9, 9, 7, 7, 6, 1, 9, 0, 4, 6, 1, 0, 3, 9, 2, 2, 7, 2, 9, 3, 9, 1, 2, 0, 8, 5, 6, 8, 5, 8, 5, 5, 0, 0, 6, 4, 6, 5, 7, 0, 8, 7, 9, 1, 6, 1, 6, 5, 8, 2, 4, 9, 9, 4, 5, 5, 5, 3, 4, 3, 0, 4, 2, 6, 9, 7, 1, 5, 3, 5, 0, 9, 7, 2, 9, 8, 4, 7, 5, 5, 2, 6, 1, 7, 2, 1, 9, 7, 5, 3, 4, 6, 2, 9, 7, 0, 3, 2, 9, 5, 6, 3, 2, 7, 3, 4, 7, 7, 1, 6, 4, 2, 5, 5, 0, 0, 0, 0, 4, 3, 0, 8, 2, 0, 0, 2, 7, 3, 8, 3, 0, 7, 6, 1, 2, 5, 5, 9, 7, 4, 7, 1, 5, 7, 5, 6, 8, 7, 9, 3, 6, 5, 8, 5, 0, 8, 4, 1, 1, 6, 7, 0, 9, 3, 6, 1, 3, 7, 5, 5, 8, 7, 3, 7, 3, 7, 4, 0, 4, 4, 1, 5, 6, 2, 2, 8, 6, 1, 7, 2, 8, 9, 6, 7, 0, 5, 0, 5, 6, 3, 7, 9, 3, 9, 1, 5, 5, 4, 6, 7, 1, 0, 7, 4, 9, 2, 7, 8, 2, 3, 4, 9, 5, 8, 8, 2, 0, 7, 1, 9, 7, 9, 5, 7, 0, 8, 8, 6, 8, 2, 6, 2, 0, 4, 0, 7, 0, 6, 0, 2, 4, 6, 5, 8, 9, 3, 9, 0, 1, 7, 6, 5, 5, 5, 8, 4, 5, 7, 0, 9, 4, 2, 3, 7, 7, 9, 1, 2, 6, 8, 5, 5, 8, 0, 9, 4, 8, 6, 9, 0, 6, 1, 5, 0, 4, 2, 6, 9, 1, 9, 6, 4, 3, 3, 0, 6, 5, 2, 2, 3, 0, 0, 4, 7, 7, 4, 3, 8, 5, 5, 6, 4, 9, 1, 2, 1, 2, 4, 2, 9, 4, 1, 9, 6, 2, 9, 4, 6, 5, 2, 7, 9, 9, 0, 9, 7, 9, 9, 0, 2, 6, 2, 8, 6, 9, 1, 9, 1, 0, 9, 4, 3, 3, 7, 4, 0, 5, 8, 1, 9, 5, 2, 7, 8, 6, 3, 5, 6, 6, 1, 6, 3, 7, 2, 6, 4, 3, 7, 4, 9, 3, 6, 6, 3, 2, 2, 9, 2, 5, 2, 3, 4, 5, 3, 6, 1, 7, 7, 5, 4, 2, 8, 5, 5, 9, 1, 6, 6, 3};
        System.out.println(checkSubarraySum(param, 2517));
    }

}
