use std::collections::HashSet;
struct Solution{}

impl Solution {
    pub fn num_identical_pairs(nums: Vec<i32>) -> i32 {
        let mut count = 0;
        let len = nums.len();
        for index in 0..len {
            for i in index + 1..len {
                if nums[i] == nums[index] {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn kids_with_candies(candies: Vec<i32>, extra_candies: i32) -> Vec<bool> {
        let kid_num = candies.len();
        let mut res:Vec<bool> = vec![true; candies.len()];
        for kid in 0..kid_num {
            let candie_num = candies[kid] + extra_candies;
            for other in 0..kid_num {
                if candie_num < candies[other] {
                    res[kid] = false;
                    break;
                }
            }
        }
        res
    }

    pub fn shuffle(nums: Vec<i32>, n: i32) -> Vec<i32> {
        let mut res: Vec<i32> = Vec::new();
        for i in 0..n {
            res.push(nums[i as usize]);
            res.push(nums[(i + n) as usize]);
        }
        res
    }

    pub fn num_jewels_in_stones(j: String, s: String) -> i32 {
        let set = j.chars().collect::<HashSet<char>>();
        s.chars().filter(|x| set.contains(x)).count() as i32
    }
}


#[cfg(test)]
mod tests {
    use super::Solution;

    #[test]
    fn code1470(){
        // 1470. 重新排列数组
        // 给你一个数组 nums ，数组中有 2n 个元素，
        // 按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
        // 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，
        // 返回重排后的数组。
        assert_eq!(
            Solution::shuffle(vec![2,5,1,3,4,7], 3),
            vec![2,3,5,4,1,7] 
        );
        assert_eq!(
            Solution::shuffle(vec![1,2,3,4,4,3,2,1], 4),
            vec![1,4,2,3,3,2,4,1]
        );
        assert_eq!(
            Solution::shuffle(vec![1,1,2,2], 2),
            vec![1,2,1,2]
        );
    }
    #[test]
    fn code1431(){
        // 1431. 拥有最多糖果的孩子
        // 给你一个数组 candies 和一个整数 extraCandies ，
        // 其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
        // 对每一个孩子，检查是否存在一种方案，
        // 将额外的 extraCandies 个糖果分配给孩子们之后，
        // 此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。

        assert_eq!(
            Solution::kids_with_candies(vec![2, 3, 5, 1, 3], 3),
            vec![true,true,true,false,true]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![4,2,1,1,2], 1),
            vec![true,false,false,false,false]
        );
        assert_eq!(
            Solution::kids_with_candies(vec![12,1,12], 10),
            vec![true,false,true]
        );
    }
    #[test]
    fn code1512() {
        assert_eq!(
            Solution::num_identical_pairs(vec![1,2,3,1,1,3]),
            4
        );
        assert_eq!(
            Solution::num_identical_pairs(vec![1,1,1,1]),
            6
        );
        assert_eq!(
            Solution::num_identical_pairs(vec![1,2,3]),
            0
        );
    }
}