package com.leetcode;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 3/6/2020 5:11 PM
 * @Author: lihongbing
 */

import java.util.*;

/**
 * 字符串相关的
 */
public class StringRelated {

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

    //5. 最长回文子串
    //给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
    //
    //示例 1：
    //输入: "babad"
    //输出: "bab"
    //注意: "aba" 也是一个有效答案。
    //示例 2：
    //输入: "cbbd"
    //输出: "bb"
    public String longestPalindrome(String s) {


        return null;
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

    //344. 反转字符串
    public void reverseString(char[] s) {
        int i = 0;
        int j = s.length - 1;
        char temp;
        while (i < j) {
            temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }

    //541. 反转字符串 II
    //给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。
    //
    //如果剩余字符少于 k 个，则将剩余字符全部反转。
    //如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
    //时间复杂度：O(n/(2k)*k)=O(n)
    //空间复杂度：O(n)
    public String reverseStr(String s, int k) {
        char[] sArray = s.toCharArray();
        int count = s.length() % (2 * k) == 0 ? s.length() / (2 * k) : s.length() / (2 * k) + 1;
        for (int i = 0; i < count; i++) {
            reverseString(sArray, i * 2 * k, i * 2 * k + k - 1);
        }
        return new String(sArray);

    }

    public static void reverseString(char[] s, int i, int j) {
        char temp;
        if (j >= s.length) j = s.length - 1;
        while (i < j) {
            temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }

    //557. 反转字符串中的单词 III
    //给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
    //输入: "Let's take LeetCode contest"
    //输出: "s'teL ekat edoCteeL tsetnoc"
    public static String reverseWords(String s) {
        char[] sArray = s.toCharArray();
        int i = 0, j = 0;
        for (; j < s.length(); j++) {
            if (sArray[j] == ' ') {
                reverseString(sArray, i, j - 1);
                i = j + 1;
            }
        }
        reverseString(sArray, i, j - 1); //最后一个单词反转
        return new String(sArray);

    }

    //58. 最后一个单词的长度
    //给定一个仅包含大小写字母和空格 ' ' 的字符串 s，返回其最后一个单词的长度。如果字符串从左向右滚动显示，那么最后一个单词就是最后出现的单词。
    //
    //如果不存在最后一个单词，请返回 0 。
    //
    //说明：一个单词是指仅由字母组成、不包含任何空格字符的 最大子字符串。
    //
    // 
    //
    //示例:
    //
    //输入: "Hello World"
    //输出: 5
    public int lengthOfLastWord(String s) {
        String s1 = s.trim();
        if (s1.length() == 0) return 0;
        int start = -1;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) == ' ') {
                start = i;
            }
        }
        return s1.length() - 1 - start;
    }

    //面试题 16.18. 模式匹配
    //你有两个字符串，即pattern和value。 pattern字符串由字母"a"和"b"组成，用于描述字符串中的模式。
    // 例如，字符串"catcatgocatgo"匹配模式"aabab"（其中"cat"是"a"，"go"是"b"），该字符串也匹配像"a"、"ab"和"b"这样的模式。
    // 但需注意"a"和"b"不能同时表示相同的字符串。编写一个方法判断value字符串是否匹配pattern字符串。
    //
    //示例 1：
    //输入： pattern = "abba", value = "dogcatcatdog"
    //输出： true
    //示例 2：
    //输入： pattern = "abba", value = "dogcatcatfish"
    //输出： false
    //示例 3：
    //输入： pattern = "aaaa", value = "dogcatcatdog"
    //输出： false
    //示例 4：
    //输入： pattern = "abba", value = "dogdogdogdog"
    //输出： true
    //解释： "a"="dogdog",b=""，反之也符合规则

    //前言
    //本题的算法实现不难，但是细节较多。题目中给出的 pattern 和 value 的长度可以为 0，因此需要充分考虑边界情况。
    //方法一：枚举
    //思路与算法
    //
    //我们设 pattern 的长度为 lp，value 的长度为 lv。根据题目描述，我们需要给字母 a 和 b 分配不同的字符串值（可以为空字符串），
    // 使得将 pattern 中的字母替换成对应的字符串后，结果与 value 相同。
    //
    //在分配字符串之前，我们不妨先分配 a 和 b 对应字符串的长度。如果确定了长度，那么我们只要将 value 按照 pattern 中出现字母的顺序，
    // 划分成 lp个子串，并判断其中 a 对应的子串是否相同，以及 b 对应的子串是否相同即可。具体地，假设 pattern 中出现了 ca个 a 以及 lp-ca个 b，
    // 并且 a 和 b 对应字符串的长度分别为 la和 lb，那么必须要满足：
    //
    //ca * la + (lp - ca) * lb = lv
    //其中 ca是已知的常量，la和 lb是未知数。这是一个二元一次方程，可能无解、有唯一解或者无数解。然而我们需要的仅仅是自然数解，也就是 la和 lb都大于等于 0 的解，
    // 因此我们可以直接枚举 la的值，它必须是 [0, lv/ca]之间的自然数，否则 lb就不会大于等于 0 了。在枚举 la之后，我们将其带入等式并解出lb。如果 lb是整数，我们就枚举了一组 a 和 b 的可能长度。
    //在枚举了长度之后，我们就可以根据 pattern 来将 value 划分成 lp个子串。
    // 具体地，我们遍历 pattern，并用一个指针 pos 来帮助我们进行划分。当我们遍历到一个 a 时，我们取出从 pos 开始，长度为 la的子串。
    // 如果这是我们第一次遇到字母 a，我们就得到了 a 应该对应的子串；否则，我们将取出的子串与 a 应该对应的子串进行比较，如果不相同，说明模式匹配失败。
    //同理，当我们遍历到一个 b 时，我们取出从 pos 开始，长度为 lb的子串，根据是否第一次遇到字母 b 来进行比较。在比较结束后，我们将 pos 向后移动，进行下一个字母的匹配。
    //
    //在遍历完成之后，如果匹配没有失败，我们还需要判断一下 a 和 b 是否对应了不同的子串。只有它们对应的子串不同时，才是一种满足题目要求的模式匹配。
    //
    //细节
    //上面的算法看上去不是很复杂：我们只需要用一重循环枚举 la ，计算出 lb，再用一重循环遍历 pattern 以及移动 pos 即可。
    // 但就像我们在「前言」部分所说的，本题有非常多的细节需要考虑。
    //
    //我们回到二元一次方程：ca * la + (lp - ca) * lb = lv
    //如果我们枚举 la，那么必须要求 ca !=0，因为在 ca = 0的情况下，原方程如果有解，那么一定有无数解（因为 la可以取任意值）。因此如果 ca = 0，我们就必须枚举lb
    //。这无疑增加了编码的复杂度，因为需要根据 ca的值选择对 la或lb进行枚举，失去了统一性。并且，如果 lp-ca也为 0，那么我们连 lb都无法枚举。
    //
    //因此，我们必须梳理一下判断的逻辑：
    //如果 pattern 为空，那么只有在 value 也为空时，它们才能匹配；
    //
    //如果 value 为空，那么如果 pattern 也为空，就和第一条的情况相同；
    //  如果 pattern 中只出现了一种字母，我们可以令该字母为空，另一没有出现的字母为任意非空串，就可以正确匹配；
    //  如果 pattern 中出现了两种字母，那么就无法正确匹配，因为这两种字母都必须为空串，而题目描述中规定它们不能表示相同的字符串；
    //如果 pattern 和 value 均非空，那么我们需要枚举 pattern 中出现的那个字母（如果两个字母均出现，可以枚举任意一个）对应的长度，使用上面提到的算法进行判断。
    //
    //对于上面的第三条，我们可以根据「对称性」减少代码的编写的复杂度：我们还是固定枚举la ，但如果 ca < lp - ca，即 a 出现的次数少于 b 出现的次数，
    // 那么我们就将 pattern 中所有的 a 替换成 b，b 替换成 a。这样做就保证了 a 出现了至少一次（ca > 0），枚举 la就不会有任何问题，同时不会影响答案的正确性。
    //
    //这样一来，我们就可以优化判断的逻辑：
    //我们首先保证 pattern 中 a 出现的次数不少于 b 出现的次数。如果不满足，我们就将 a 和 b 互相替换；
    //如果 value 为空，那么要求 pattern 也为空（lp = 0）或者只出现了字母 a（lp - ca = 0），这两种情况均等同于 lp - ca = 0。在其余情况下，都无法匹配成功；
    //如果 pattern 为空且 value 不为空，那么无法匹配成功；
    //如果 pattern 和 value 均非空，我们就可以枚举 la并使用上面提到的算法进行判断。

    //本题的时空复杂度不易分析（因为涉及到二元一次方程解的个数），这里只近似地给出一个结果。
    //时间复杂度：O(lv^2)，其中 lp和 lv分别是 pattern 和 value 的长度。由于 la必须是 [0, lv/ca] 中的自然数，并且 1/2lp <=ca<=lp，因此方程解的个数为 O(lv/lp)。
    // 对于每一组解，我们需要 O(lp + lv)的时间来进行判断，因此总时间复杂度为 O((lp + lv)*lv/lp)。根据大 O 表示法的定义（渐进上界），可以看成 O(lv^2)  。
    //
    //空间复杂度：O(lv)。我们需要存储 a 和 b 对应的子串，它们的长度之和不会超过 lv。
    public boolean patternMatching(String pattern, String value) {
        int count_a = 0, count_b = 0;
        for (char ch : pattern.toCharArray()) {
            if (ch == 'a') {
                ++count_a;
            } else {
                ++count_b;
            }
        }
        if (count_a < count_b) {
            int temp = count_a;
            count_a = count_b;
            count_b = temp;
            char[] array = pattern.toCharArray();
            for (int i = 0; i < array.length; i++) {
                array[i] = array[i] == 'a' ? 'b' : 'a';
            }
            pattern = new String(array);
        }
        if (value.length() == 0) {
            return count_b == 0;
        }
        if (pattern.length() == 0) {
            return false;
        }
        for (int len_a = 0; count_a * len_a <= value.length(); ++len_a) {
            int rest = value.length() - count_a * len_a;
            if ((count_b == 0 && rest == 0) || (count_b != 0 && rest % count_b == 0)) {
                int len_b = (count_b == 0 ? 0 : rest / count_b);
                int pos = 0;
                boolean correct = true;
                String value_a = "", value_b = "";
                for (char ch : pattern.toCharArray()) {
                    if (ch == 'a') {
                        String sub = value.substring(pos, pos + len_a);
                        if (value_a.length() == 0) {
                            value_a = sub;
                        } else if (!value_a.equals(sub)) {
                            correct = false;
                            break;
                        }
                        pos += len_a;
                    } else {
                        String sub = value.substring(pos, pos + len_b);
                        if (value_b.length() == 0) {
                            value_b = sub;
                        } else if (!value_b.equals(sub)) {
                            correct = false;
                            break;
                        }
                        pos += len_b;
                    }
                }
                if (correct && !value_a.equals(value_b)) {
                    return true;
                }
            }
        }
        return false;
    }

    //方法二：
    //说实在工作中真碰到这样的问题，第一反应是用现有的工具解决。算法可能要折腾半小时，而且看了众多解答也没有觉得特别的简洁。
    //
    //根据 pattern 构造一个正则它不香吗？
    //
    //aabb -> (\w*)\1(\w*)\2
    //abba -> (\w*)(\w*)\2\1
    //上面 \1 和 \2 表示对前面分组的反向引用。

}
