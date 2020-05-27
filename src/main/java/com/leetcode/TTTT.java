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

    //9. 回文数
    //输入: 121
    //输出: true
    public boolean isPalindrome(int x) {
        char[] str = String.valueOf(x).toCharArray();
        int length = str.length;
        for (int i = 0; i < length / 2; i++) {
            if (str[i] != str[length - i - 1]) return false;
        }
        return true;
    }

    //125. 验证回文串
    //给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
    //说明：本题中，我们将空字符串定义为有效的回文串。
    //输入: "A man, a plan, a canal: Panama"
    //输出: true
    //A~Z = 65~90 , a~z = 97~122 , 0~9 = 48~57
    //思想，也很简单，遍历一遍就行，不符合条件的指针向前或后走一步，然后继续比较
    public static boolean isPalindrome(String s) {
        for (int i = 0, j = s.length() - 1; i < j; ) {
            int t1 = convert(s.charAt(i));
            int t2 = convert(s.charAt(j));
            if (t1 == 0) {
                i++;
                continue;
            }
            if (t2 == 0) {
                j--;
                continue;
            }
            if (t1 != t2) return false;
            i++;
            j--;

        }
        return true;
    }

    public static int convert(char t) {
        if (t >= 'a' && t <= 'z') {
            return t;
        } else if (t >= 'A' && t <= 'Z') {
            return t + 32;//转换成小写
        } else if (t >= '0' && t <= '9') {
            return t;
        } else {
            return 0;
        }
    }


    //680. 验证回文字符串 Ⅱ
    //给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
    //利用首尾指针先找到字符不一样的位置记为i,j
    //然后分两种情况，i向后一步或者j向前一步，再继续对比，只要有一个成功就是成功
    //时间复杂度：O(n)
    //空间复杂度：O(1)
    public static boolean validPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;

        int m = 0;
        int n = 0;
        for (; i < j; i++, j--) {
            if (s.charAt(i) != s.charAt(j)) {
                break;
            }
        }

        //分两种情况，i向后一步或者j向前一步，然后继续对比，只要有一个成功就成功
        boolean res1 = true;
        boolean res2 = true;

        for (m = i + 1, n = j; m < n; m++, n--) {
            if (s.charAt(m) != s.charAt(n)) {
                res1 = false;
                break;
            }
        }
        if (res1) return true;
        for (m = i, n = j - 1; m < n; m++, n--) {
            if (s.charAt(m) != s.charAt(n)) {
                res2 = false;
                break;
            }
        }
        return res2;


    }


    public static String convert(String s, int numRows) {
        if (numRows == 1) return s;
        //定义numRow维数组
        String[] res = new String[numRows];
        for (int i = 0; i < res.length; i++) {
            res[i] = "";
        }
        int j = 0;
        int m = 0;
        for (int i = 0; i < s.length(); i++) {
            if (i % (numRows - 1) == 0) {
                m++;
            }
            if (m % 2 != 0) {
                res[j] += (s.charAt(i));
                j++;
            } else {
                res[j] += (s.charAt(i));
                j--;
            }
        }

        String result = "";
        for (int n = 0; n < numRows; n++) {
            System.out.println(res[n]);
            result += (res[n]);
        }
        return result;
    }

//字符          数值
//I             1
//V             5
//X             10
//L             50
//C             100
//D             500
//M             1000
    //I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
    //X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
    //C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。

    public static int romanToInt(String s) {
        Map map = new HashMap<Character, Integer>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        int result = 0;
        char cur, last = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            cur = s.charAt(i);
            if (((last == 'V' || last == 'X') && cur == 'I')
                    || ((last == 'L' || last == 'C') && cur == 'X')
                    || ((last == 'D' || last == 'M') && cur == 'C')
            ) {
                result -= Integer.parseInt(map.get(cur).toString());
            } else {
                result += Integer.parseInt(map.get(cur).toString());
            }
            last = cur;
        }

        return result;


    }

    private int getValue(char ch) {
        switch (ch) {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;
        }
    }

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


    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = dummyHead;
        int cur = 0;
        while (l1 != null || l2 != null) {
            if (l1 == null) {
                cur = l2.val;
                l2 = l2.next;
            } else if (l2 == null) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val < l2.val) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val >= l2.val) {
                cur = l2.val;
                l2 = l2.next;
            }
            p.next = new ListNode(cur);
            p = p.next;
        }
        return dummyHead.next;
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

    //242. 有效的字母异位词
    //法1.通过将 s 的字母重新排列成 t 来生成变位词。因此，如果 t 是 s 的变位词，对两个字符串进行排序将产生两个相同的字符串。
    // 此外，如果 s 和 t 的长度不同，t 不能是 s 的变位词，我们可以提前返回。
    //时间复杂度：O(nlogn)，假设 n 是 s 的长度，排序成本 O(nlogn) 和比较两个字符串的成本 O(n)。排序时间占主导地位，总体时间复杂度为O(nlogn)。
    //空间复杂度：O(1)，空间取决于排序实现，如果使用 heapsort，通常需要 O(1)辅助空间。注意，在 Java 中，toCharArray() 制作了一个字符串的拷贝，所以它花费 O(n) 额外的空间，但是我们忽略了这一复杂性分析，因为：
    //  这依赖于语言的细节。
    //  这取决于函数的设计方式。例如，可以将函数参数类型更改为 char[]。
    public static boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        char[] str1 = s.toCharArray();
        char[] str2 = t.toCharArray();
        Arrays.sort(str1);
        Arrays.sort(str2);
        return Arrays.equals(str1, str2);
    }

    //242. 有效的字母异位词
    //法2.哈希表
    // 为了检查 t 是否是 s 的重新排列，我们可以计算两个字符串中每个字母的出现次数并进行比较。
    // 因为 s 和 t 都只包含 a-z的字母，所以一个简单的 26 位计数器表就足够了。
    //我们需要两个计数器数表进行比较吗？实际上不是，因为我们可以用一个计数器表计算 s 字母的频率，用 t 减少计数器表中的每个字母的计数器，
    // 然后检查计数器是否回到零。
    public static boolean isAnagram2(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        int[] counter = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counter[s.charAt(i) - 'a']++;
            counter[t.charAt(i) - 'a']--;
        }
        for (int count : counter) {
            if (count != 0) {
                return false;
            }
        }
        return true;
    }

    //49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        //思想：把数组strs里面的每一个string先排序，然后把整个字符串数组string排序，最后就相当于是split一下，
        //有一个很大的问题：最后出来的结果内容中的字符串不是原先的字符串了，所以这个思想有问题！
//        String[] strs1 = new String[strs.length];
//        for (int i = 0; i < strs.length; i++) {
//            char[] temp = strs[i].toCharArray();
//            Arrays.sort(temp);
//            strs1[i] = String.valueOf(temp);
//        }
//        Arrays.sort(strs1);
//        List<List<String>> result = new ArrayList<List<String>>();
//        String last = null;
//        int j = -1;
//        for (int i = 0; i < strs1.length; i++) {
//            String cur = strs1[i];
//            if (!cur.equals(last)) {
//                List<String> list = new ArrayList<String>();
//                result.add(list);
//                list.add(cur);
//                j++;
//            } else {
//                result.get(j).add(cur);
//            }
//            last = cur;
//        }
        //思想，用一个辅助数组，一次遍历，但是确定当前扫描到的字符串放到那个分组里面，需要用到查找
        //时间复杂度：遍历每个字符串，每个字符串都需要排序，还得查找分组位置（如果用hashmap，这个时间是常数），与官方的方法一类似，它使用了hashmap稍微简洁点，
        //时间复杂度：O(NKlogK)，其中 N 是 strs 的长度，而 K 是 strs 中字符串的最大长度。当我们遍历每个字符串时，外部循环具有的复杂度为 O(N)。
        // 然后，我们在 O(KlogK) 的时间内对每个字符串排序。
        //空间复杂度：O(NK)，排序存储在 result 中的全部信息内容。
        List<List<String>> result = new ArrayList<List<String>>();
        List<String> aux = new ArrayList<>();
        for (int i = 0; i < strs.length; i++) {
            String cur = strs[i];
            char[] chars = cur.toCharArray();
            Arrays.sort(chars);
            String temp = String.valueOf(chars);
            int m = aux.indexOf(temp);
            int n = m;
            if (m == -1) {
                aux.add(temp);
                n = aux.size() - 1;
            }
            if (n >= result.size()) {
                List<String> list = new ArrayList<>();
                result.add(list);
                list.add(cur);
            } else {
                result.get(n).add(cur);
            }
        }
        return result;
    }


    //438. 找到字符串中所有字母异位词
    public static List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.length() - i < p.length()) break;

            String temp = s.substring(i, i + p.length());

            //判断temp和p是否是字母异位词即可
            int[] counter = new int[26];
            for (int j = 0; j < p.length(); j++) {
                counter[p.charAt(j) - 'a']++;
                counter[temp.charAt(j) - 'a']--;
            }
            boolean flag = true;
            for (int count : counter) {
                if (count != 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                result.add(i);
            }
        }

        return result;
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


    //206. 反转链表
    public static ListNode reverseList(ListNode head) {
        //方法一：增加两个指针，遍历的时候保留临时指针
        //时间复杂度：O(n)，假设 n 是列表的长度，时间复杂度是 O(n)。
        //空间复杂度：O(1)
        ListNode pre = null;
        ListNode next = head;
        while (next != null) {
            ListNode temp = next.next; //必须
            next.next = pre;
            pre = next;
            next = temp;
        }
        return pre;

    }

    //206. 反转链表
    public ListNode reverseList2(ListNode head) {
        //方法二：递归思想：https://leetcode-cn.com/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-by-leetcode/
        //递归比较难以理解，看评论：看了半个小时可算是把这个递归看懂了！
        // 不妨假设链表为1，2，3，4，5。按照递归，当执行reverseList（5）的时候返回了5这个节点，reverseList(4)中的p就是5这个节点，
        // 我们看看reverseList（4）接下来执行完之后，5->next = 4, 4->next = null。这时候返回了p这个节点，也就是链表5->4->null，
        // 接下来执行reverseList（3），代码解析为4->next = 3,3->next = null，这个时候p就变成了，5->4->3->null, reverseList(2),
        // reverseList(1)依次类推，p就是:5->4->3->2->1->null
        //时间复杂度：O(n)，假设 nn 是列表的长度，那么时间复杂度为 O(n)。
        //空间复杂度：O(n)，由于使用递归，将会使用隐式栈空间。递归深度可能会达到 n 层。
        if (head == null || head.next == null) return head;
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    //92. 反转链表 II      反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
    public static ListNode reverseBetween(ListNode head, int m, int n) {
        //思想：计数器，将要反转的部分截取出来反转，然后在拼回去
        //比如输入 1 -> 2 -> 3 -> 4 -> 5 -> null     2,4
        //        n1   p         q    n2
        //首先扫描一遍链表，利用p,q记录m和n对应的位置，n1和n2记录m-1和n+1的位置
        //然后将q.next=null，针对p这个子链表进行反转，最后再拼接回去
        //时间复杂度：第一次会扫描一遍整个链表，然后会扫描一遍子链表[m,n] ，O(l+n-m),其实达不到题目所要求的只扫描一遍的要求。
        int i = 0;
        ListNode cur = head;
        ListNode p = head;
        ListNode q = null;

        ListNode n1 = null;
        ListNode n2 = null;

        while (cur != null) {
            i++;
            if (i == m - 1) n1 = cur;
            if (i == m) p = cur; //肯定不为空
            if (i == n) q = cur; //肯定不为空
            if (i == n + 1) n2 = cur;
            cur = cur.next;
        }


        q.next = null;
        //将p反转，同题目206
        ListNode pre = null;
        ListNode next = p;

        while (next != null) {
            ListNode temp = next.next;
            next.next = pre;
            pre = next;
            next = temp;
        }

        if (n1 != null) {
            n1.next = pre;
        } else {
            head = pre;
        }
        p.next = n2;


        return head;
    }
    ////思路：head表示需要反转的头节点，pre表示需要反转头节点的前驱节点
    // 同时我们也需要设置一个哑节点dummy，因为m=1时，我们可以也有前驱节点，剩余部分看代码即可。
    //    //我们需要反转n-m次，我们将head的next节点移动到需要反转链表部分的首部，需要反转链表部分剩余节点依旧保持相对顺序即可
    //反转的过程中，有点像头插法。
    //    //比如1->2->3->4->5,m=1,n=5
    //    //第一次反转：1(head) 2(next) 3 4 5 反转为 2 1 3 4 5
    //    //第二次反转：2 1(head) 3(next) 4 5 反转为 3 2 1 4 5
    //    //第三次发转：3 2 1(head) 4(next) 5 反转为 4 3 2 1 5
    //    //第四次反转：4 3 2 1(head) 5(next) 反转为 5 4 3 2 1
    //    ListNode* reverseBetween(ListNode* head, int m, int n) {
    //        ListNode *dummy=new ListNode(-1);
    //        dummy->next=head;
    //        ListNode *pre=dummy;
    //        for(int i=1;i<m;++i)pre=pre->next;
    //        head=pre->next;
    //        for(int i=m;i<n;++i){
    //            ListNode *nxt=head->next;
    //            //head节点连接nxt节点之后链表部分，也就是向后移动一位
    //            head->next=nxt->next;
    //            //nxt节点移动到需要反转链表部分的首部
    //            nxt->next=pre->next;
    //            pre->next=nxt;//pre继续为需要反转头节点的前驱节点
    //        }
    //        return dummy->next;
    //    }

    //234. 回文链表    请判断一个链表是否为回文链表。要求空间复杂度达到O(1)
    //输入: 1->2->2->1
    //输出: true
    //方法一，最简单的方法是，将值复制到数组中后用双指针法
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    //方法二，递归，空间复杂度还是O(n)，太复杂了，可以看官网
    //方法三，将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较。比较完成后我们应该将链表恢复原样。
    // 虽然不需要恢复也能通过测试用例，因为使用该函数的人不希望链表结构被更改。
    //          空间复杂度O(1)
    public static boolean isPalindrome(ListNode head) {
        if (head == null) return true;

        // Find the end of first half and reverse second half.
        ListNode firstHalfEnd = Utils.endOfFirstHalf(head); //通过快慢指针法找到前半部分的尾节点
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // Check whether or not there is a palindrome.
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) result = false;
            p1 = p1.next;
            p2 = p2.next;
        }

        // Restore the list and return the result.
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    //136. 只出现一次的数字     给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
    //解法一：暴力查找
    //两次循环，每次从数组中取一个数，记为cur，然后从剩下的数中查找，如果找不到，则cur即为要找的那个数。这种解法时间复杂度是 O(n^2)
    //解法二：排序
    //使用快排，复杂度 O(nlogn)
    //解法三：
    //利用 Hash 表，Time: O(n) Space: O(n)
    //        Map<Integer, Integer> map = new HashMap<>();
    //        for (Integer i : nums) {
    //            Integer count = map.get(i);
    //            count = count == null ? 1 : ++count;
    //            map.put(i, count);
    //        }
    //        for (Integer i : map.keySet()) {
    //            Integer count = map.get(i);
    //            if (count == 1) return i;
    //        }
    //        return -1; // can't find it.
    //解法五：使用集合Hashset存储数组中出现的所有数字，并计算数组中的元素之和。由于集合保证元素无重复，因此计算集合中的所有元素之和的两倍，即为每个元素出现两次的情况下的元素之和。由于数组中只有一个元素出现一次，其余元素都出现两次，因此用集合中的元素之和的两倍减去数组中的元素之和，剩下的数就是数组中只出现一次的数字。
    //2×(a+b+c)−(a+a+b+b+c)=c ，时间复杂度：O(n),空间复杂度O(n)
    //解法四：异或，牛逼的解法，善于题目中的已有信息！！！！, 时间复杂度O(n),空间复杂度O(1)
    public static int singleNumber(int[] nums) {
        int ans = nums[0];
        if (nums.length > 1) {
            for (int i = 1; i < nums.length; i++) {
                ans = ans ^ nums[i];
            }
        }
        return ans;
    }

    //137. 只出现一次的数字 II
    //给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
    //循环查找，hashmap，可以做到，但是时间复杂度不行
    //官方解法，用复杂的位计算，比较复杂，暂时没看
    public static int singleNumber137(int[] nums) {

        return 0;
    }


    //268. 缺失数字
    //给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
    //解法一，先快速排序，然后遍历
    //时间复杂度：O(nlgn),由于排序的时间复杂度为 O(nlogn)，扫描数组的时间复杂度为 O(n)，因此总的时间复杂度为 O(nlogn)。
    //空间复杂度：O(1) 或 O(n)。空间复杂度取决于使用的排序算法，根据排序算法是否进行原地排序（即不使用额外的数组进行临时存储）
    public static int missingNumber(int[] nums) {
        Arrays.sort(nums);
        int last = nums[0];
        if (last != 0) return 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] - last != 1) return last + 1;
            last = nums[i];
        }
        return last + 1;
        //解法2，Hash表， 将所有数子插入到hash表中，然后遍历0～n,看哪个不存在
        //时间复杂度：O(n)。集合的插入操作的时间复杂度都是 O(1)，一共插入了 n 个数，时间复杂度为O(n)。
        //      集合的查询操作的时间复杂度同样是 O(1)，最多查询 n+1 次，时间复杂度为 O(n)。因此总的时间复杂度为 O(n)。
        //空间复杂度：O(n)。集合中会存储 n 个数，因此空间复杂度为O(n)。
        //方法三：位运算^   异或满足结合律
        //时间复杂度：O(n)。这里假设异或运算的时间复杂度是常数的，总共会进行 O(n)次异或运算，因此总的时间复杂度为 O(n)。
        //空间复杂度：O(1)。算法中只用到了 O(1) 的额外空间，用来存储答案。
        //方法四：数学公式 0+1+2+...n=n(n+1)/2,减去数组中所有数字的和就是缺失的数字
        //时间复杂度：O(n)。求出数组中所有数的和的时间复杂度为 O(n)，高斯求和公式的时间复杂度为 O(1)，因此总的时间复杂度为 O(n)。
        //空间复杂度：O(1)。算法中只用到了 O(1) 的额外空间，用来存储答案。

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

    //560. 和为K的子数组
    //方法一：暴力法
    //时间复杂度：O(n^2)
    //空间复杂度：O(1)
    public static int subarraySum(int[] nums, int k) {
        int m = 0;
        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == k) {
                    m++;
                }
            }
        }
        return m;
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
    //                                  i
    //                                 pre[6]-k=0
    //              则j=1，2，4都是符合的
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
                    count += mp.get(key); //这句话改成return true就不报超时了，可能如果是count += mp.get(key)，时间复杂度会变成O(n^2)力扣后台的限制了
                }
            }
            mp.put(prepre, mp.getOrDefault(prepre, 0) + 1);
        }
        return count > 0;
    }

    //下面是官网的解答，也是基于前缀和，但是改的很优雅
    public static boolean checkSubarraySum2(int[] nums, int k) {
        int sum = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (k != 0)
                sum = sum % k;
            if (map.containsKey(sum)) {
                if (i - map.get(sum) > 1)
                    return true;
            } else
                map.put(sum, i); //把索引put进去
        }
        return false;
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


    //974. 和可被 K 整除的子数组
    public static int subarraysDivByK(int[] A, int K) {
        //我的方法，前缀和
        int[] sum = new int[A.length];
        int temp = 0;
        int count = 0;
        for (int i = 0; i < A.length; i++) {
            temp += A[i];
            sum[i] = temp;
        }

        for (int i = A.length - 1; i >= 0; i--) {
            for (int j = -1; j < i; j++) {
                if (j == -1) {
                    if ((sum[i] - 0) % K == 0) count++;
                } else {
                    if ((sum[i] - sum[j]) % K == 0) count++;

                }
            }

        }
        return count;

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
