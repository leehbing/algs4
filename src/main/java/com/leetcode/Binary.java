package com.leetcode;

import java.math.BigInteger;
import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 1/6/2020 10:42 AM
 * @Author: lihongbing
 */
public class Binary {
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
    //方法一：hashmap，可以做到，但是时间复杂度不行
    //
    //
    //方法二：借助于hashset： 3×(a+b+c)−(a+a+a+b+b+b+c)=2c
    //    Set<Long> set = new HashSet<>();
    //    long sumSet = 0, sumArray = 0;
    //    for(int n : nums) {
    //      sumArray += n;
    //      set.add((long)n);
    //    }
    //    for(Long s : set) sumSet += s;
    //    return (int)((3 * sumSet - sumArray) / 2);
    //时间复杂度：O(N)
    //空间复杂度：O(N),hashset存储N/3个元素的集合
    //方法三，用复杂的位计算，比较复杂，暂时没看
    //∼x表示位运算 NOT
    //
    //x&y表示位运算 AND
    //
    //x⊕y表示位运算 XOR    用于检测出现奇数次的位：1、3、5 等
    //      0⊕x=x(出现一次)       x⊕x=0       x⊕x⊕x=x(出现三次)
    //      可以检测出出现一次的位和出现三次的位，但是要注意区分这两种情况
    //为了区分出现一次的数字和出现三次的数字，使用两个位掩码：seen_once 和 seen_twice。
    //思路是：
    //      仅当 seen_twice 未变时，改变 seen_once。
    //      仅当 seen_once 未变时，改变seen_twice。
    //最后位掩码 seen_once 仅保留出现一次的数字，不保留出现三次的数字。
    //====》有点难绕
    public static int singleNumber137(int[] nums) {
        int seenOnce = 0, seenTwice = 0;

        for (int num : nums) {
            // first appearence:
            // add num to seen_once
            // don't add to seen_twice because of presence in seen_once

            // second appearance:
            // remove num from seen_once
            // add num to seen_twice

            // third appearance:
            // don't add to seen_once because of presence in seen_twice
            // remove num from seen_twice
            seenOnce = ~seenTwice & (seenOnce ^ num);
            seenTwice = ~seenOnce & (seenTwice ^ num);
        }
        return seenOnce;

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
        //        int missing = nums.length;
        //        for (int i = 0; i < nums.length; i++) {
        //            missing ^= i ^ nums[i];
        //        }
        //        return missing;
        //
        //方法四：数学公式 0+1+2+...n=n(n+1)/2,减去数组中所有数字的和就是缺失的数字
        //时间复杂度：O(n)。求出数组中所有数的和的时间复杂度为 O(n)，高斯求和公式的时间复杂度为 O(1)，因此总的时间复杂度为 O(n)。
        //空间复杂度：O(1)。算法中只用到了 O(1) 的额外空间，用来存储答案。

    }

    //421. 数组中两个数的最大异或值
    //给定一个非空数组，数组中元素为 a0, a1, a2, … , an-1，其中 0 ≤ ai < 231 。
    //
    //找到 ai 和aj 最大的异或 (XOR) 运算结果，其中0 ≤ i,  j < n 。  你能在O(n)的时间解决这个问题吗？
    //输入: [3, 10, 5, 25, 2, 8]
    //输出: 28
    //解释: 最大的结果是 5 ^ 25 = 28
    //题目要求 O(N)O(N) 时间复杂度，下面会讨论两种典型的 O(N)O(N) 复杂度解法。
    //
    //利用哈希集合存储按位前缀
    //利用字典树存储按位前缀
    //这两种解法背后的思想是一样的，都是先将整数转化成二进制形式，再从最左侧的比特位开始逐一处理来构建最大异或值。两个方法的不同点在于采用了不同的数据结构来存储按位前缀。第一个方法在给定的测试集下执行速度更快，但第二种方法更加普适，更加简单。
    //
    //方法一：利用哈希集合存储按位前缀  见官方详解
    public static int findMaximumXOR(int[] nums) {
        //复杂度分析
        //时间复杂度：O(N)
        //空间复杂度：O(1)
        int maxNum = nums[0];
        for (int num : nums) maxNum = Math.max(maxNum, num);
        // length of max number in a binary representation
        int L = (Integer.toBinaryString(maxNum)).length();

        int maxXor = 0, currXor;
        Set<Integer> prefixes = new HashSet<>();
        for (int i = L - 1; i > -1; --i) {
            // go to the next bit by the left shift
            maxXor <<= 1;
            // set 1 in the smallest bit
            currXor = maxXor | 1;
            prefixes.clear();
            // compute all possible prefixes
            // of length (L - i) in binary representation
            for (int num : nums) prefixes.add(num >> i);
            // Update maxXor, if two of these prefixes could result in currXor.
            // Check if p1^p2 == currXor, i.e. p1 == currXor^p2.            a^b=c那么a=b^c
            for (int p : prefixes) {
                if (prefixes.contains(currXor ^ p)) {
                    maxXor = currXor;
                    break;
                }
            }
        }
        return maxXor;

    }

    //方法二：逐位字典树  见官方详解
    public int findMaximumXOR2(int[] nums) {
        // Compute length L of max number in a binary representation
        int maxNum = nums[0];
        for (int num : nums) maxNum = Math.max(maxNum, num);
        int L = (Integer.toBinaryString(maxNum)).length();

        // zero left-padding to ensure L bits for each number
        int n = nums.length, bitmask = 1 << L;
        String[] strNums = new String[n];
        for (int i = 0; i < n; ++i) {
            strNums[i] = Integer.toBinaryString(bitmask | nums[i]).substring(1);
        }

        TrieNode trie = new TrieNode();
        int maxXor = 0;
        for (String num : strNums) {
            TrieNode node = trie, xorNode = trie;
            int currXor = 0;
            for (Character bit : num.toCharArray()) {
                // insert new number in trie
                if (node.children.containsKey(bit)) { //构建字典树
                    node = node.children.get(bit);
                } else {
                    TrieNode newNode = new TrieNode();
                    node.children.put(bit, newNode);
                    node = newNode;
                }

                // compute max xor of that new number
                // with all previously inserted
                Character toggledBit = bit == '1' ? '0' : '1';
                if (xorNode.children.containsKey(toggledBit)) {
                    currXor = (currXor << 1) | 1;
                    xorNode = xorNode.children.get(toggledBit);
                } else {
                    currXor = currXor << 1;
                    xorNode = xorNode.children.get(bit);
                }
            }
            maxXor = Math.max(maxXor, currXor);
        }

        return maxXor;
    }

    class TrieNode {
        HashMap<Character, TrieNode> children = new HashMap<Character, TrieNode>();

        public TrieNode() {
        }
    }


    //187. 重复的DNA序列
    //编写一个函数来查找目标子串，目标子串的长度为 10，且在 DNA 字符串 s 中出现次数超过一次。
    //输入：s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    //输出：["AAAAACCCCC", "CCCCCAAAAA"]
    //方法一：线性时间窗口切片 + HashSet
    //时间复杂度：O((N-L)L)
    //空间复杂度：O((N-L)L)
    public List<String> findRepeatedDnaSequences(String s) {
        int L = 10, n = s.length();
        HashSet<String> seen = new HashSet(), output = new HashSet();

        // iterate over all sequences of length L
        for (int start = 0; start < n - L + 1; ++start) {
            String tmp = s.substring(start, start + L);
            if (seen.contains(tmp)) output.add(tmp);
            seen.add(tmp);
        }
        return new ArrayList<String>(output);
    }

    //318. 最大单词长度乘积
    public int maxProduct(String[] words) {

        return 0;
    }

    //67. 二进制求和
    public static String addBinary(String a, String b) {
        //方法一：将 a 和 b 转换为十进制整数。求和。将求和结果转换为二进制整数。
        //算法的时间复杂度为 O(N+M)，但是该方法存在两个问题。
        //
        //在 Java 中，该方法受输入字符串 a 和 b 的长度限制。字符串长度太大时，不能将其转换为 Integer，Long 或者 BigInteger 类型。
        //33 位 1，不能转换为 Integer。
        //65 位 1，不能转换为 Long。
        //500000001 位 1，不能转换为 BigInteger。

        //return Integer.toBinaryString(Integer.parseInt(a, 2) + Integer.parseInt(b, 2));


        //方法一：逐位计算    一种古老的经典算法，无需把数字转换成十进制，直接逐位计算和与进位即可
        //初始进位 carry=0，如果数字 a 的最低位是 1，则将 1 加到进位carry；同理如果数字 b 的最低位是 1，则也将 1 加到进位。
        // 此时最低位有三种情况：(00) (01) (11)
        //
        //然后将 carry 的最低位作为最低位的值，将 carry 的最高位移至下一位继续计算。
        //重复上述步骤，直到数字 a 和 b 的每一位计算完毕。最后如果carry 的最高位不为 0，则将最高位添加到计算结果的末尾。
        // 最后翻转结果得到求和结果。
        int n = a.length(), m = b.length();
        if (n < m) return addBinary(b, a);
        int L = Math.max(n, m);

        StringBuilder sb = new StringBuilder();
        int carry = 0, j = m - 1;
        for (int i = L - 1; i > -1; --i) {
            if (a.charAt(i) == '1') ++carry;
            if (j > -1 && b.charAt(j--) == '1') ++carry;
            if (carry % 2 == 1) {
                sb.append('1');
            } else {
                sb.append('0');
            }
            carry /= 2;
        }
        if (carry == 1) sb.append('1');
        sb.reverse();

        return sb.toString();

    }

    //方法二：位操作
    //a=            1 1 1 1
    //b=            0 0 1 0
    //a^b=          1 1 0 1    ===>异或的结果就是两个数字无进位相加的结果
    //(a&b)<<1    0 0 1 0 0    ===>需要的进位
    //所以问题可以转化为：首先计算两个数字的无进位相加结果（异或）x以及进位y， 然后再将x和y相加，不断进行下去，直到进位y=0
    public static String addBinary2(String a, String b) {
        BigInteger x = new BigInteger(a, 2); //二进制
        BigInteger y = new BigInteger(b, 2);
        BigInteger zero = new BigInteger("0", 2);
        BigInteger carry, answer;
        while (y.compareTo(zero) != 0) {
            answer = x.xor(y);
            carry = x.and(y).shiftLeft(1);
            x = answer;
            y = carry;
        }
        return x.toString(2);

    }


    public static void main(String[] args) {


    }
}
