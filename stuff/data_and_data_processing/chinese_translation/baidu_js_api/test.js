const translate = require("baidu-translate-api");
var arg = process.argv[2];
translate(arg).then(res => {
    console.log(res.trans_result.dst);
    // Let's translate it!
});
