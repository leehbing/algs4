package com.leetcode;

import java.util.Stack;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 22/6/2020 5:33 PM
 * @Author: lihongbing
 */
public class StackRelated {
    //20. 有效的括号
    //给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
    //
    //有效字符串需满足：
    //
    //左括号必须用相同类型的右括号闭合。
    //左括号必须以正确的顺序闭合。
    //注意空字符串可被认为是有效字符串。
    //
    //示例 1:
    //输入: "()"
    //输出: true
    //示例 2:
    //输入: "()[]{}"
    //输出: true
    //示例 3:
    //输入: "(]"
    //输出: false
    //示例 4:
    //输入: "([)]"
    //输出: false
    //示例 5:
    //输入: "{[]}"
    //输出: true
    //方法一：栈
    //其实可以增加一个判断：如果栈的深度大于字符串长度的1/2，就返回false。因为当出现这种情况的时候，即使后面的全部匹配，栈也不会为空。
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
}
