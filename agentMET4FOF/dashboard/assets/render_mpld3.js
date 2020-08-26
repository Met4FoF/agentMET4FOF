function render_mpld3(fig_id, fig_content){
        function mpld3_load_lib(url, callback){
          var s = document.createElement('script');
          s.src = url;
          s.async = true;
          s.onreadystatechange = s.onload = callback;
          s.onerror = function(){console.warn("failed to load library " + url);};
          document.getElementsByTagName("head")[0].appendChild(s);
        }
        if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
           // already loaded: just create the figure
           !function(mpld3){
               d3.select("#"+fig_id).selectAll("*").remove();
               mpld3.draw_figure(fig_id, fig_content)
           }(mpld3);
        }else if(typeof define === "function" && define.amd){
           // require.js is available: use it to load d3/mpld3
           require.config({paths: {d3: "https://d3js.org/d3.v5"}});
           require(["d3"], function(d3){
              window.d3 = d3;
              mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.1.js", function(){
                 d3.select("#"+fig_id).selectAll("*").remove();
                 mpld3.draw_figure(fig_id, fig_content)
              });
            });
        }else{
            // require.js not available: dynamically load d3 & mpld3
            mpld3_load_lib("https://d3js.org/d3.v5.js", function(){
                 mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.5.1.js", function(){
                     d3.select("#"+fig_id).selectAll("*").remove();
                       mpld3.draw_figure(fig_id, fig_content)
                    })
                 });
        }

}

if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
    render_mpld3_each: function(value){
//        console.log(value)
        for (mpld3_children of value[0].props.children) {
            if ("id" in mpld3_children.props && mpld3_children.type == "Div") {
                fig_id = mpld3_children.props.id
                if (fig_id.includes("d3_"))
                {
                fig_content = JSON.parse(mpld3_children.props.children.props.children)
                render_mpld3(fig_id, fig_content)
                }
                }
        }
        return 0
    }
}

