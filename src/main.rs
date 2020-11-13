fn main(){
    let mut v = vec![1, 2, 3];
    println!("{:?}", v.drain(..));
    // Drain([1, 2, 3])

    let mut s = String::from("codeleet");
    let s = s.drain(..);
    println!("{:?}", s);
    // 得到Drain { .. }   

    let s = s.enumerate();
    println!("{:?}", s);
    // Enumerate { iter: Drain { .. }, count: 0 }

    let s = s.collect::<Vec<(usize,char)>>();
    println!("{:?}", s);
    // [(usize, char), (usize, char)]


    // use std::collections::HashMap;
    // let mut m: HashMap<u32, &str> = HashMap::new();
    // m.insert(1, "hello");
    // m.insert(2, "world");
    // let v: Vec<u32> = vec![1,2,3];
    // let b: Vec<(u32, &str)> = v 
    //     .iter()
    //     .filter_map(|&k| m.get(&k).map(|&v| (k, v)))
    //     .collect();
    // for (x,y) in b {
    //     println!("{},{}", x,y);
    // }


}