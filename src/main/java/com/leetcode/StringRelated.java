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
}
